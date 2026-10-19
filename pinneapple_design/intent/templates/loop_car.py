"""Toy car through a loop-the-loop: track radius, clothoid entry, launch speed, car mass, track strip.

Physics: rigid-body dynamics of a small car on a curved track with contact (normal force >= 0), rolling resistance
and aerodynamic drag. Models: energy balance (closed form), the 1-D along-track ODE, and NVIDIA Newton (contact +
Coulomb friction of a rigid capsule on the track mesh) as the full check. Failure modes: the car leaves the track at
the top (normal force < 0) or stalls; the track strip yields under the peak contact load; the loop is taller than
the space available.

Unknown friction and drag are not guessed: the design must work over a stated *envelope* (rolling-resistance
coefficient up to ``crr_max``, drag coefficient over the bluff-to-streamlined range, launcher scatter), and the
build instructions say how to measure the real values.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from .. import materials as M
from ..physics import loop_track as LT
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

TRACK_MATERIALS = ("pla_4043d", "al6061_t6")
# the car of the "toy" scale: a 1:64 die-cast class body; ballast sets the mass
CAR = {"length": 0.070, "width": 0.030, "height": 0.025, "wheel_r": 0.006, "wheel_w": 0.003}
ENVELOPE = ("assumed design envelope, not a measured value: the design must work for any value in the range; "
            "measure yours (build step 1)")


def _rho_air() -> float:
    from pinneapple_physics.closed_form.plate_fin_heatsink import air_properties
    return air_properties(20.0)["rho"]


class LoopCarTemplate(ProblemTemplate):
    name = "loop_car"
    title = "Toy car loop-the-loop"
    domain = "rigid-body dynamics"
    keywords = {"loop": 1.5, "loop-the-loop": 2.0, "looping": 1.5, "toy car": 1.5, "car": 0.5, "track": 0.5,
                "hot wheels": 2.0, "marble": 0.5, "upside down": 0.5, "clothoid": 1.0}
    physics = ("rigid-body dynamics of the car along the track (Newton's second law along the path)",
               "unilateral contact: normal force N = m (v^2 kappa + g cos theta) >= 0",
               "rolling resistance C_rr N and aerodynamic drag 0.5 rho C_d A v^2",
               "clothoid (Euler spiral) entry and exit: curvature rising linearly to 1/R",
               "track strip in bending under the peak contact load (simply supported between stand clamps)")
    assumptions = ("the car is a point on its centre-of-mass path for the 1-D model; Newton models it as a rigid "
                   "sphere of the wheel radius",
                   "rolling losses are lumped into one coefficient C_rr = force / normal force",
                   "wheels: four solid PLA discs, their rotation adds inertia (not weight)",
                   "air at 20 °C, 1 atm (density from the ideal gas law)",
                   "the track is rigid and does not move (clamped to its stands)")
    slots = (
        Slot("loop_radius", "loop radius", "m", "cm", ("radius", "loop of", "loop radius"),
             bounds=(0.05, 0.40), role="scale"),
        Slot("loop_diameter", "loop diameter", "m", "cm", ("diameter", "wide loop", "loop diameter"),
             bounds=(0.10, 0.80), role="scale"),
        Slot("car_mass", "car mass", "kg", "g", ("car weighs", "weighs", "mass", "weight", "heavy", "car of"),
             bounds=(0.01, 0.30), role="scale"),
        Slot("launch_speed", "launch speed", "m/s", "m/s", ("launch", "launched at", "speed", "starts at"),
             bounds=(0.5, 8.0), role="load"),
        Slot("max_height", "space available (height)", "m", "cm", ("tall", "high", "height", "fit under", "fits"),
             bounds=(0.1, 2.0), role="limit"),
        Slot("crr_max", "rolling-resistance coefficient (worst case)", "", "", ("rolling resistance", "crr",
                                                                               "rolling friction"),
             default=0.05, default_source=ENVELOPE, bounds=(0.0, 0.3), role="environment"),
        Slot("cd_range", "drag coefficient (worst case)", "", "", ("drag coefficient", "cd"),
             default=1.2, default_source="worst case of the bluff-body range; a cube in cross-flow is about 1.05 "
                                         "(Hoerner, Fluid-Dynamic Drag, 1965)", bounds=(0.1, 2.0)),
        Slot("launch_scatter", "launcher speed scatter (+-)", "", "%", ("scatter", "repeatability",
                                                                         "launcher tolerance"),
             default=0.05, default_source=ENVELOPE, bounds=(0.0, 0.3)),
        Slot("track_material", "track material", "choice", keywords=("track material", "printed", "made of"),
             default="pla_4043d", default_source="template default: printable on a desktop 3-D printer",
             choices=TRACK_MATERIALS, role="material"),
        Slot("support_spacing", "distance between track clamps", "m", "cm", ("clamp", "support every",
                                                                              "supports every"),
             default=0.10, default_source="template default (build step: clamp the strip every 10 cm)",
             bounds=(0.02, 0.5)),
        Slot("safety_factor", "safety factor on track yield", "", "", ("safety factor", "factor of safety"),
             default=2.0, default_source="template default (engineering convention for a known static load; "
                                         "state yours)", bounds=(1.0, 10.0)),
    )
    objectives = (
        ObjectiveChoice("launch_speed", "min", "launch speed", ("gentle launch", "slow", "low speed", "easy")),
        ObjectiveChoice("g_peak", "min", "peak load on the car (g)", ("smooth", "gentle", "comfortable",
                                                                       "low g", "g-load")),
        ObjectiveChoice("loop_radius", "max", "loop radius", ("big loop", "large loop", "biggest", "impressive")),
        ObjectiveChoice("footprint", "min", "track length on the table", ("compact", "small", "short")),
    )
    output_units = {"v_top": "m/s", "g_peak": "g", "loop_height": "m", "footprint": "m", "track_stress_mpa": "MPa",
                    "track_mass_kg": "kg", "launch_speed": "m/s", "loop_radius": "m", "thickness": "m"}
    output_notes = {"v_top": "speed at the top, worst-case envelope", "g_peak": "peak N / (m g), worst-case envelope",
                    "track_stress_mpa": "strip bending stress under the peak contact load"}

    def default_objectives(self):
        return [self.objectives[0], self.objectives[1]]

    failure_modes = (
        FailureMode("margin_contact", "car stays on the track",
                    "minimum over the loop of N / (m g) at the worst-case envelope (< 0: the car falls at the top "
                    "or stalls)"),
        FailureMode("margin_track", "track strip does not yield",
                    "yield / (safety factor x bending stress under the peak contact load) - 1"),
        FailureMode("margin_clearance", "car body clears the track in the loop",
                    "wheel radius / sag of the wheelbase chord b^2 / (8 R) - 1 (< 0: the body scrapes the track)"),
        FailureMode("margin_height", "loop fits the space available",
                    "space available / loop height - 1 (no limit stated: not constraining)", critical=False),
    )
    fidelities = (
        Fidelity(CLOSED_FORM, "energy balance", "v_top^2 = v0^2 - 2 g z_top - losses at the mean speed; "
                                                "contact at the top if v_top^2 >= g R"),
        Fidelity(REDUCED, "1-D along-track ODE", "d(v^2)/ds with gravity, N(s), rolling resistance and drag, RK4, "
                                                 "vectorised over designs"),
        Fidelity(FULL, "Newton rigid-body contact", "NVIDIA Newton XPBD: the car on the swept 3-D track mesh "
                                                     "(rails, helical exit), two time steps", modules=("newton", "warp")),
    )

    # ------------------------------------------------------------------ scope
    def check_scope(self, spec: RequirementSpec) -> list[str]:
        low = spec.text.lower()
        out = []
        for w in ("roller coaster", "rollercoaster", "passenger", "rider", "people", "person", "full size",
                  "full-size", "real car", "motorbike", "bicycle"):
            if w in low:
                out.append(f"'{w}': carrying people needs amusement-ride standards (ASTM F2291 acceleration limits, "
                           "restraints, structural codes) and a ride engineer; this template designs toy-scale "
                           "tracks only")
                break
        return out

    # ------------------------------------------------------------------ variables
    def _fixed(self, spec: RequirementSpec) -> dict[str, float]:
        f = {}
        if spec.get("loop_radius") is not None:
            f["radius"] = spec["loop_radius"]
        elif spec.get("loop_diameter") is not None:
            f["radius"] = 0.5 * spec["loop_diameter"]
        if spec.get("car_mass") is not None:
            f["mass"] = spec["car_mass"]
        if spec.get("launch_speed") is not None:
            f["v0"] = spec["launch_speed"]
        return f

    def _all(self) -> list[DesignVariable]:
        return [DesignVariable("radius", 0.08, 0.30, "m", "loop radius R"),
                DesignVariable("clothoid_ratio", 0.0, 2.5, "", "clothoid length / R"),
                DesignVariable("v0", 1.0, 6.0, "m/s", "launch speed"),
                DesignVariable("mass", 0.02, 0.15, "kg", "car mass")]

    def variables(self, spec: RequirementSpec) -> list[DesignVariable]:
        fixed = self._fixed(spec)
        return [v for v in self._all() if v.name not in fixed]

    def _unpack(self, X: np.ndarray, spec: RequirementSpec) -> dict[str, np.ndarray]:
        X = np.atleast_2d(X)
        fixed = self._fixed(spec)
        vs = self.variables(spec)
        out = {v.name: X[:, j] for j, v in enumerate(vs)}
        for k, v in fixed.items():
            out[k] = np.full(len(X), v)
        return out

    def full_design(self, design: dict[str, Any], spec: RequirementSpec) -> dict[str, Any]:
        """The design variables plus the fixed values and the sized track strip."""
        d = dict(design)
        for k, v in self._fixed(spec).items():
            d.setdefault(k, v)
        d["clothoid_length"] = d["clothoid_ratio"] * d["radius"]       # [m]
        d["material"] = spec["track_material"]
        y = self.evaluate(REDUCED, np.array([[design[v.name] for v in self.variables(spec)]]), spec)
        d["thickness"] = float(y["thickness"][0])
        return d

    @staticmethod
    def strip_thickness(F, span, sy, sf):
        """Thinnest strip (0.5 mm steps, at least 1.5 mm to print) with yield / (sf sigma) >= 1 for a point load
        ``F`` at mid-span between clamps ``span`` apart: sigma = 1.5 F span / (w t^2), w = 40 mm."""
        t = np.sqrt(1.5 * F * span * sf / (0.04 * sy))
        return np.maximum(np.ceil(t / 5e-4 - 1e-9) * 5e-4, 1.5e-3)

    @staticmethod
    def wheel_inertia() -> float:
        """sum I / r^2 of four solid PLA discs (I = m r^2 / 2): 2 x the mass of the four wheels [kg]."""
        r, w = CAR["wheel_r"], CAR["wheel_w"]
        m1 = M.get("pla_4043d")["density"] * math.pi * r * r * w
        return 4 * 0.5 * m1

    # ------------------------------------------------------------------ models
    def evaluate(self, level: str, X: np.ndarray, spec: RequirementSpec, *, crr=None, cd=None, v_scale=None,
                 m_scale=None, com_height=None, wheel_inertia=None, run_out=0.3, thickness=None, **_: Any
                 ) -> dict[str, np.ndarray]:
        p = self._unpack(X, spec)
        n = len(next(iter(p.values())))
        R, c, v0, m = p["radius"], p["clothoid_ratio"], p["v0"], p["mass"]
        Lc = c * R
        # the worst case of the envelope unless the robustness study passes sampled values
        crr_in = crr
        crr = np.full(n, spec["crr_max"]) if crr is None else np.asarray(crr, float)
        cd = np.full(n, spec["cd_range"]) if cd is None else np.asarray(cd, float)
        vs = np.full(n, 1 - spec["launch_scatter"]) if v_scale is None else np.asarray(v_scale, float)
        ms = np.ones(n) if m_scale is None else np.asarray(m_scale, float)
        h = 0.5 * CAR["height"] if com_height is None else com_height
        mw = self.wheel_inertia() if wheel_inertia is None else wheel_inertia
        A = CAR["width"] * CAR["height"]
        rho = _rho_air()
        kw = dict(crr=crr, cd=cd, area=A, com_height=h, wheel_inertia=mw, rho=rho, run_in=0.3)
        if level == CLOSED_FORM:
            r = LT.closed_form(R, Lc, v0 * vs, m * ms, **kw)
            contact = r["top_margin"]
            n_peak = r["n_max"]
            v_top = r["v_top"]
            z_top = r["z_top"]
        elif level == REDUCED:
            run = LT.along_track(R, Lc, v0 * vs, m * ms, run_out=run_out, **kw)
            loop_len = R * 2 * np.pi
            frac = np.where(np.isfinite(run.s_fail), (run.s_fail - 0.3) / np.maximum(loop_len, 1e-9), 1.0)
            contact = np.where(run.completes, run.n_min, -1.0 + np.clip(frac, 0, 0.99))
            n_peak, v_top = run.n_max, run.v_top
            z_top = LT.closed_form(R, Lc, v0, m, crr=0, cd=0, com_height=h)["z_top"]
        else:
            raise ValueError(f"level {level!r}: the full model runs through verify()")
        # track strip: point load F = n_peak m g at mid-span between clamps, simply supported. The strip is sized at
        # the design point (worst-case envelope); a Monte-Carlo sample keeps that size and checks its own load.
        mat = M.get(spec["track_material"])
        sy, sf, span = mat["yield_strength"], spec["safety_factor"], spec["support_spacing"]
        F = np.maximum(n_peak, 1.0) * m * ms * LT.G0
        if thickness is None:
            yd = n_peak if (crr_in is None and v_scale is None) else self.evaluate(level, X, spec)["g_peak"]
            thickness = self.strip_thickness(np.maximum(yd, 1.0) * m * LT.G0, span, sy, sf)
        t = np.broadcast_to(np.asarray(thickness, float), (n,))
        sigma = 1.5 * F * span / (0.04 * t ** 2)
        margin_track = sy / (sf * sigma) - 1.0
        # body clearance: the wheelbase chord sags into the curve by b^2 / (8 R); the body must clear the track
        b = 0.64 * CAR["length"]
        margin_clear = CAR["wheel_r"] / (b ** 2 / (8 * R)) - 1.0
        height = z_top + 0.02
        hmax = spec.get("max_height")
        margin_h = hmax / height - 1.0 if hmax else np.ones(n)
        th_c = 0.5 * Lc / R
        _, x_cl = LT._clothoid_rise(R, Lc)
        x_hi = np.maximum(0.3 + x_cl + R * (1 - np.sin(th_c)), 0.6 + 2 * x_cl - 2 * R * np.sin(th_c))
        x_lo = np.minimum(0.0, 0.3 + x_cl - R * (1 + np.sin(th_c)))
        footprint = x_hi - x_lo
        rho_s = mat["density"]
        L_track = 0.6 + 2 * Lc + R * (2 * np.pi - 2 * th_c)
        return {"margin_contact": contact, "margin_track": margin_track, "margin_clearance": margin_clear,
                "margin_height": margin_h, "thickness": t,
                "launch_speed": v0, "g_peak": n_peak, "loop_radius": R, "footprint": footprint, "v_top": v_top,
                "loop_height": height, "track_stress_mpa": sigma / 1e6,
                "track_mass_kg": rho_s * ((0.04 + 2 * t) * t + 2 * 0.008 * t) * L_track}

    # ------------------------------------------------------------------ robustness
    def uncertainties(self, spec: RequirementSpec, design: dict[str, Any]) -> list[Uncertain]:
        cmax, cdmax, sc = spec["crr_max"], spec["cd_range"], spec["launch_scatter"]
        return [Uncertain("crr", 0.5 * cmax, 0.5 * cmax, "uniform", "", f"0 to {cmax:g}: {ENVELOPE}"),
                Uncertain("cd", 0.5 * (0.3 + cdmax), 0.5 * (cdmax - 0.3), "uniform", "",
                          f"0.3 (streamlined) to {cdmax:g} (bluff)"),
                Uncertain("v_scale", 1.0, sc / 2, "normal", "", f"launcher scatter +-{100 * sc:.0f} % taken as 2 "
                                                                "standard deviations"),
                Uncertain("m_scale", 1.0, 0.05, "uniform", "", "car mass within +-5 % of target (weigh it)")]

    # ------------------------------------------------------------------ full fidelity
    def verify(self, design: dict[str, Any], spec: RequirementSpec, workdir: str | None = None) -> FullResult:
        """Newton on the 3-D track at two time steps, frictionless (its friction on curved meshes is not usable, see
        the notes), judged against the worst-case envelope; plus Newton's own analytic sliding-friction case."""
        from pinneapple_simulation.external_solvers.newton_physics import validate

        d = self.full_design(design, spec)
        R, Lc, v0, m = d["radius"], d["clothoid_ratio"] * d["radius"], d["v0"], d["mass"]
        track = self.track(d)
        rc = CAR["wheel_r"]
        same = LT.along_track(R, Lc, v0, m, crr=0.0, cd=0.0, com_height=rc, run_in=0.3, run_out=0.3)
        v = np.maximum(same.profile["v"], 0.05)
        T = float(np.sum(np.diff(same.profile["s"]) / (0.5 * (v[1:] + v[:-1])))) * 1.15
        dts = (5e-5, 2.5e-5)
        runs = [LT.newton_check(track, m, v0, dt=dt, duration=T, radius=rc) for dt in dts]
        fine = runs[-1]
        vt = np.array([r.v_top for r in runs])
        v_same = float(same.v_top[0])
        fric = validate(solver="xpbd", cases=["sliding_friction"], log=None)[0]
        checks = [
            {"name": "car stays on the track and reaches the end (both time steps)", "critical": True,
             "passed": all(r.completes for r in runs), "value": float(max(r.max_gap for r in runs)),
             "limit": 2 * rc, "detail": "largest gap between the car and the running surface in the loop [m]; "
                                        "the car must not leave the track before the end of the run-out"},
            {"name": "speed at the top insensitive to the time step", "critical": False,
             "passed": bool(abs(vt[1] / vt[0] - 1) <= 0.05), "value": abs(vt[1] / vt[0] - 1), "limit": 0.05,
             "detail": f"v_top = {vt.round(4).tolist()} m/s at dt = {list(dts)} s"},
            {"name": "Newton within 10 % of the 1-D model without losses", "critical": False,
             "passed": bool(abs(vt[-1] / v_same - 1) <= 0.10), "value": abs(vt[-1] / v_same - 1), "limit": 0.10,
             "detail": f"v_top {vt[-1]:.3f} (Newton, frictionless) vs {v_same:.3f} m/s (1-D, no friction, no air): "
                       "the gap is the contact's numerical dissipation"},
            {"name": "Newton Coulomb friction (analytic sliding case)", "critical": False,
             "passed": bool(fric["passed"]), "value": fric["error"], "limit": fric["tolerance"],
             "detail": f"{fric['quantity']}: {fric['simulated']:.4g} vs {fric['reference']:.4g} on a flat surface"},
        ]
        e0 = 0.5 * v0 ** 2
        z_top = float(fine.pos[:, 2].max())
        out = {"v_top": float(vt[-1]), "completes": float(all(r.completes for r in runs)),
               "max_gap_m": float(fine.max_gap), "v_top_1d_no_losses": v_same,
               "newton_energy_lost": float(1 - (0.5 * vt[-1] ** 2 + LT.G0 * (z_top - rc - 1e-4)) / e0)}
        notes = [f"NVIDIA Newton {fine.meta.get('newton_version')}, XPBD, {fine.meta.get('iterations')} iterations: "
                 f"the car is a sphere of the wheel radius ({rc * 1e3:.0f} mm) on the 3-D track mesh (strip, rails, "
                 "the exit shifted sideways), launched at the design speed; no air",
                 f"Newton loses {100 * out['newton_energy_lost']:.1f} % of the launch energy by the top with no "
                 "friction at all: XPBD contact dissipation on the curved mesh, which barely changes with the time "
                 "step or a finer mesh. The run is therefore judged against the worst-case envelope (it must still "
                 "go round and keep at least the envelope's speed at the top), not for equality.",
                 "Friction is not used in the loop run: Newton's Coulomb friction is checked on its analytic flat "
                 "sliding case, but on the curved track mesh a rolling sphere with friction loses most of its energy "
                 "(measured while building this template), so the rolling-resistance envelope stays in the 1-D "
                 "model."]
        if workdir:
            os.makedirs(workdir, exist_ok=True)
            np.savez_compressed(os.path.join(workdir, "newton_loop.npz"), t=fine.t, pos=fine.pos, vel=fine.vel,
                                s=fine.s, v_top=vt, dt=np.array(dts))
        return FullResult(out, checks, FULL, self.fidelity(FULL).label, float(sum(r.wall_s for r in runs)),
                          {"trajectory": fine, "track": track, "runs": runs}, notes, compare={"v_top": "v_top"},
                          compare_kind={"v_top": "at_least"})

    # ------------------------------------------------------------------ package
    def track(self, d: dict[str, Any]) -> LT.LoopTrack:
        return LT.LoopTrack(d["radius"], d["clothoid_ratio"] * d["radius"], run_in=0.3, run_out=0.3, width=0.04,
                            pitch=0.05)

    def geometry(self, spec: RequirementSpec, design: dict[str, Any]):
        from pinneapple_design.geometry_library.core import Body
        from pinneapple_design.geometry_library.kit import merge, tf

        d = self.full_design(design, spec)
        tr = self.track(d)
        parts = LT.track_meshes(tr, thickness=d["thickness"], rail=0.008, n=500)
        colour = "plastic" if d["material"] == "pla_4043d" else "aluminium"
        bodies = [Body(f"track_{k}", V, F, colour) for k, (V, F) in parts.items()]
        body, wheels = LT.car_mesh(CAR["length"], CAR["width"], CAR["height"], CAR["wheel_r"])
        bodies.append(Body("car_body", *tf(body, t=(0.08, 0.0, 0.0)), "paint", group="car"))
        bodies.append(Body("car_wheels", *tf(merge(wheels), t=(0.08, 0.0, 0.0)), "grey", group="car",
                           closed=True))
        return bodies

    def bill_of_materials(self, spec: RequirementSpec, design: dict[str, Any]) -> list[dict[str, Any]]:
        d = self.full_design(design, spec)
        tr = self.track(d)
        mat = M.get(d["material"])
        L = tr.length()
        t = d["thickness"]
        vol = L * ((0.04 + 2 * t) * t + 2 * 0.008 * t)
        pla = M.get("pla_4043d")
        return [
            {"item": "track strip with side rails", "material": mat.label, "quantity": 1,
             "size": f"{L * 100:.0f} cm long, 40 mm wide running surface, {d['thickness'] * 1e3:.1f} mm thick, "
                     "8 mm rails", "mass_g": round(vol * mat["density"] * 1e3, 1),
             "properties": {k: {"value": mat.props[k].value, "unit": mat.props[k].unit, "source": mat.cite(k)}
                            for k in ("density", "yield_strength")}},
            {"item": "car (body + ballast)", "material": "any; ballast to the target mass", "quantity": 1,
             "size": f"{CAR['length'] * 1e3:.0f} x {CAR['width'] * 1e3:.0f} x {CAR['height'] * 1e3:.0f} mm",
             "mass_g": round(d["mass"] * 1e3, 1), "properties": {}},
            {"item": "wheels", "material": pla.label, "quantity": 4,
             "size": f"{2 * CAR['wheel_r'] * 1e3:.0f} mm diameter, {CAR['wheel_w'] * 1e3:.0f} mm wide",
             "mass_g": round(self.wheel_inertia() / 2 * 1e3, 2),
             "properties": {"density": {"value": pla["density"], "unit": "kg/m^3", "source": pla.cite("density")}}},
            {"item": "stand clamps", "material": "any", "quantity": int(math.ceil(L / spec["support_spacing"])),
             "size": f"one every {spec['support_spacing'] * 100:.0f} cm", "mass_g": None, "properties": {}},
        ]

    def operating_window(self, spec: RequirementSpec, design: dict[str, Any], ctx: dict[str, Any]
                         ) -> list[dict[str, Any]]:
        """Measured on the 1-D model: the ranges over which the chosen design still completes the loop."""
        d = self.full_design(design, spec)
        R, Lc, v0, m = d["radius"], d["clothoid_ratio"] * d["radius"], d["v0"], d["mass"]
        h, mw, A, rho = 0.5 * CAR["height"], self.wheel_inertia(), CAR["width"] * CAR["height"], _rho_air()
        base = dict(crr=spec["crr_max"], cd=spec["cd_range"], area=A, com_height=h, wheel_inertia=mw, rho=rho)

        v_grid = np.linspace(0.3, 12.0, 2400)
        r = LT.along_track(R, Lc, v_grid, m, **base)
        good = r.completes & (r.n_min >= 0)
        v_min = float(v_grid[np.argmax(good)]) if good.any() else float("nan")
        # upper launch speed: track strip margin (peak load grows with speed)
        mat = M.get(d["material"])
        sigma = 1.5 * np.maximum(r.n_max, 1) * m * LT.G0 * spec["support_spacing"] / (0.04 * d["thickness"] ** 2)
        strong = mat["yield_strength"] / (spec["safety_factor"] * sigma) >= 1.0
        both = good & strong
        v_max = float(v_grid[np.flatnonzero(both)[-1]]) if both.any() else float("nan")
        capped = bool(both.any() and both[-1])
        # largest rolling-resistance coefficient at the design launch speed (other envelopes at worst case)
        crr_grid = np.linspace(0.0, 0.3, 601)
        rc = LT.along_track(R, Lc, v0 * (1 - spec["launch_scatter"]), m, **{**base, "crr": crr_grid})
        okc = rc.completes & (rc.n_min >= 0)
        crr_ok = float(crr_grid[np.flatnonzero(okc)[-1]]) if okc.any() else float("nan")
        m_grid = np.linspace(0.01, 0.3, 291)
        rm = LT.along_track(R, Lc, v0 * (1 - spec["launch_scatter"]), m_grid, **base)
        okm = rm.completes & (rm.n_min >= 0)
        m_lo = float(m_grid[np.argmax(okm)]) if okm.any() else float("nan")
        m_hi = float(m_grid[np.flatnonzero(okm)[-1]]) if okm.any() and not okm[-1] else None
        th = 0.5 * Lc / R
        return [
            {"quantity": "launch speed", "value": round(v0, 3), "min": round(v_min, 3),
             "max": None if capped else round(v_max, 3), "unit": "m/s", "fidelity": "1-D along-track ODE",
             "note": f"min: still completes with C_rr = {spec['crr_max']:g} and C_d = {spec['cd_range']:g}; max: "
                     + ("no modelled failure mode up to 12 m/s (the car leaving the run-out fast is not modelled)"
                        if capped else "track strip reaches yield / safety factor")},
            {"quantity": "car mass", "value": round(m * 1e3, 1), "min": round(m_lo * 1e3, 1),
             "max": None if m_hi is None else round(m_hi * 1e3, 1), "unit": "g", "fidelity": "1-D along-track ODE",
             "note": "at the design launch speed minus the launcher scatter, worst-case envelope"
                     + ("; no upper limit up to 300 g (heavier cars lose less to drag)" if m_hi is None else "")},
            {"quantity": "wheel rolling-resistance coefficient", "value": None, "min": 0.0, "max": round(crr_ok, 4),
             "unit": "-", "fidelity": "1-D along-track ODE",
             "note": "largest C_rr that still completes at the design launch speed minus the scatter"},
            {"quantity": "loop radius (top arc)", "value": round(R * 100, 2), "min": None, "max": None, "unit": "cm",
             "fidelity": "design", "note": "radius of the circular arc at the top"},
            {"quantity": "clothoid entry length", "value": round(Lc * 100, 2), "min": None, "max": None,
             "unit": "cm", "fidelity": "design",
             "note": f"curvature 0 -> 1/R; turns the track by {math.degrees(th):.1f} deg (same at the exit)"},
        ]

    def build_instructions(self, spec: RequirementSpec, design: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
        d = self.full_design(design, spec)
        tr = self.track(d)
        mat = M.get(d["material"])
        win = {w["quantity"]: w for w in ctx.get("window", [])}
        crr_ok = win.get("wheel rolling-resistance coefficient", {}).get("max")
        steps = [
            "Measure the car's rolling resistance first: on a straight strip of the track material tilted by an "
            "angle a, find the smallest angle at which the car keeps rolling at constant speed; C_rr = tan(a). "
            + (f"This design completes the loop for C_rr up to {crr_ok:g}." if crr_ok is not None else ""),
            f"Print or cut the track strip ({mat.label}, {d['thickness'] * 1e3:.1f} mm thick) from track.stl / "
            f"track.step: {tr.length() * 100:.0f} cm of running surface, 40 mm wide, with 8 mm side rails.",
            f"Shape the loop: top arc radius {d['radius'] * 100:.1f} cm, entry and exit clothoids "
            f"{d['clothoid_length'] * 100:.1f} cm long; the exit is shifted 5 cm sideways so it clears the "
            "entry.",
            f"Clamp the strip to a stand every {spec['support_spacing'] * 100:.0f} cm; the loop is "
            f"{tr.height() * 100:.1f} cm tall.",
            f"Ballast the car to {d['mass'] * 1e3:.0f} g (+-5 %).",
            f"Set the launcher to {d['v0']:.2f} m/s (+-{100 * spec['launch_scatter']:.0f} %); check with a phone "
            "slow-motion video over the 30 cm run-in.",
            "First runs: start 10 % below the design speed and raise it until the car completes the loop; the "
            "operating window table gives the speed range that works.",
        ]
        return steps

    def limits(self, spec: RequirementSpec) -> list[str]:
        return ["The car is a point on its centre-of-mass path (1-D model) or a sphere (Newton); wheel bounce, "
                "steering and track joints are not modelled.",
                "Newton's check has no friction and no air, and its contact dissipates energy: it confirms the 3-D "
                "track lets the car go round with margin, not the 1-D model's speeds to a tolerance.",
                "Rolling resistance and drag are envelopes, not measurements: the design covers them, a car outside "
                "them (bent axle, dirty wheels) may not complete the loop.",
                "The track strength check is a beam under a point load between clamps; the clamps, the stand and "
                "fatigue of the strip are not checked.",
                "Toy scale only: nothing here applies to rides that carry people."]

    def figures(self, spec: RequirementSpec, design: dict[str, Any], ctx: dict[str, Any], out_dir: str
                ) -> dict[str, str]:
        from ..figures import loop_figures
        return loop_figures(self, spec, self.full_design(design, spec), ctx, out_dir)
