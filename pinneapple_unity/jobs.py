"""The jobs a Unity scene can ask PINNeAPPle to compute, and the checks that come back with each field.

A job turns a part's vertices plus its parameters into one or more per-vertex fields over the scene's time steps,
**and** the physics checks of that field -- the point of the bridge is not that a number arrives in Unity, it is
that the number arrives with its boundary conditions, its residual and its decision attached.

Two jobs ship with the package (both closed-form, both checked):

``slab_transient``
    transient 1-D conduction along a part (:func:`pinneapple_physics.closed_form.heat.slab_transient`, Carslaw &
    Jaeger) -> ``temperature``; checked with the heat-equation residual, the Dirichlet walls, the maximum principle
    on the exported field, and the analytic bound on the approach to the steady profile.
``cantilever_stress``
    bending stress along a part loaded at its tip (:mod:`pinneapple_physics.closed_form.cantilever_fatigue`) ->
    ``bending_stress`` (MPa), optionally ``fatigue_life_cycles``; checked against statics recomputed from first
    principles, the linear moment line, the stress-free tip, and the yield / target-life decisions.

More jobs are three lines: :func:`register_job` a function ``JobRequest -> JobResult``; the bridge and the ingest
pipeline pick it up by name (``pinneapple_unity.register_job("my_job", fn)``).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .protocol import local_coordinate

#: The tolerance the trust-layer checks use for a field sampled from geometry (relative to the field's range).
_REL = 2e-2


@dataclass
class JobRequest:
    """What a job is given: the part, its vertices, the scene's time steps and the parameters from the link."""

    job: str
    part: str
    vertices: np.ndarray
    times: list[float] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    faces: np.ndarray | None = None

    @property
    def t(self) -> np.ndarray:
        """The time steps in seconds, starting at 0 (a twin's first step is its initial state)."""
        ts = np.asarray(self.times or [0.0], float)
        if ts.size == 0:
            ts = np.zeros(1)
        return ts - ts[0]


@dataclass
class JobResult:
    """A job's answer: fields per part, the checks of those fields, and the resulting decision."""

    part: str
    job: str
    fields: dict[str, tuple[np.ndarray, str]] = field(default_factory=dict)  # name -> ((T, V), unit)
    checks: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def decision(self) -> str:
        """``APPROVED`` (every check ran and passed), ``REJECT`` (a critical check failed) or ``REVIEW``."""
        ran = [c for c in self.checks if c.get("passed") is not None]
        if any(c["passed"] is False and c.get("critical", True) for c in ran):
            return "REJECT"
        if any(c["passed"] is False for c in ran):
            return "REVIEW"
        return "APPROVED" if ran and len(ran) == len(self.checks) else "REVIEW"

    def report(self) -> dict[str, Any]:
        return {"part": self.part, "job": self.job, "decision": self.decision, "checks": self.checks,
                "notes": self.notes, "detail": self.detail}


def _check(name: str, passed: bool | None, kind: str, value: float | None = None,
           threshold: float | None = None, detail: str = "", critical: bool = True) -> dict[str, Any]:
    return {"name": name, "passed": None if passed is None else bool(passed), "kind": kind,
            "value": None if value is None else float(value),
            "threshold": None if threshold is None else float(threshold), "detail": detail,
            "critical": bool(critical)}


def _range_relative(err: float, values: np.ndarray) -> float:
    """``err`` as a fraction of the field's own spread (so a check reads the same for K and for degrees)."""
    spread = float(np.ptp(values)) or max(float(np.abs(values).max()), 1e-12)
    return float(err) / spread


# ----------------------------------------------------------------------------------------------- slab transient
def slab_transient_job(req: JobRequest) -> JobResult:
    """Transient conduction along a part: ``T(x, t)`` from the closed-form slab solution.

    Parameters (``link.jobs[i].params``) -- all optional except ``alpha`` when the part is not mild steel:

    ``alpha`` thermal diffusivity [m^2/s], default 1e-5 (the order of a steel; a parameter, not a material claim)
    ``t_left``/``t_right`` the two wall temperatures [C] (default 80 / 20), ``t0`` the initial one [C] (default 20)
    ``length`` the span the profile is laid over [m] (default: the part's bounding box along ``axis``)
    ``axis`` ``"x"``/``"y"``/``"z"`` or ``"auto"`` (default: the longest side of the part)
    """
    from pinneapple_physics.closed_form.heat import slab_transient

    p = dict(req.params)
    alpha = float(p.get("alpha", 1e-5))
    t_left = float(p.get("t_left", 80.0))
    t_right = float(p.get("t_right", 20.0))
    t0 = float(p.get("t0", 20.0))
    u, L = local_coordinate(req.vertices, str(p.get("axis", "auto")), p.get("length"), bool(p.get("flip", False)))
    times = req.t

    values = slab_transient(u[None, :], times[:, None], L, alpha, t_left, t_right, t0)
    res = JobResult(req.part, req.job, {"temperature": (np.asarray(values, dtype=np.float32), "C")})

    # 1-2. the heat-equation residual and the walls, on a regular (x, t) grid with the trust layer's own card.
    #      Two things about that grid. It starts after the initial-condition jump (at t = 0 the analytic field is
    #      discontinuous at both walls, which no finite-difference stencil represents), and its time steps are uniform:
    #      the residual is a second-order central difference in t, and a scene whose steps jump from 60 s to 3600 s
    #      would be judged on that jump. The card therefore covers the smooth window [t_check, t_end], not the scene's
    #      own (possibly logarithmic) steps; the exported field is the scene's.
    t_check = 0.05 * L * L / alpha
    t_end = max(float(times[-1]), 4.0 * t_check)
    grid_t = np.linspace(t_check, t_end, 25)
    grid_x = np.linspace(0.0, L, max(41, 2 * int(np.sqrt(len(u))) + 1))
    grid_T = slab_transient(grid_x[:, None], grid_t[None, :], L, alpha, t_left, t_right, t0)
    card = None
    try:
        import pinneapple as pp
        card = pp.trust_card({"x": grid_x, "t": grid_t}, grid_T, pde="heat", alpha=alpha,
                             dirichlet={"x_min": t_left, "x_max": t_right})
        for c in card.checks:
            if c.passed is None and c.name.startswith(("reference", "applicability")):
                continue                       # no reference data and no trained model: not applicable, not skipped
            res.checks.append(_check(c.name, c.passed, getattr(c, "kind", "physics"),
                                     None if c.value is None else float(c.value),
                                     None if c.threshold is None else float(c.threshold), getattr(c, "detail", "")))
    except Exception as exc:  # noqa: BLE001 - a trust card that cannot run is reported, never silently dropped
        res.notes.append(f"the trust card could not run ({exc}); the field is exported unverified")
        res.checks.append(_check("trust card", None, "physics", detail=f"could not run: {exc}"))

    # 3. maximum principle on the field that is actually exported, every step.
    lo, hi = min(t_left, t_right, t0), max(t_left, t_right, t0)
    over = _range_relative(max(float(lo - values.min()), float(values.max() - hi), 0.0), values)
    res.checks.append(_check("maximum principle (no value outside the initial and wall range)", over <= _REL, "physics",
                             over, _REL, f"range [{values.min():.4g}, {values.max():.4g}] C inside [{lo:g}, {hi:g}] C"))

    # 4. the approach to the steady profile is bounded by the first Fourier mode, exactly:
    #    |T(x, t) - T_steady(x)| <= A1 exp(-alpha pi^2 t / L^2), A1 = 2/pi max|T0 - T_steady|.
    steady = t_left + (t_right - t_left) * u / L
    dev = float(np.abs(values[-1] - steady).max())
    a1 = 2.0 / np.pi * float(np.abs(t0 - steady).max())
    bound = a1 * float(np.exp(-alpha * np.pi ** 2 * float(times[-1]) / (L * L)))
    res.checks.append(_check("approach to the steady profile within the first-mode bound", dev <= 1.05 * bound + 1e-9,
                             "reference", dev, 1.05 * bound,
                             f"|T(x, t_end) - T_steady|_max = {dev:.4g} C, first-mode bound A1 exp(-alpha pi^2 t/L^2)"
                             f" = {bound:.4g} C (A1 = {a1:.4g} C)"))
    res.detail.update({"length_m": L, "alpha_m2_s": alpha, "t_left_c": t_left, "t_right_c": t_right,
                       "t0_c": t0, "diffusion_time_s": L * L / alpha, "u_range_m": [float(u.min()), float(u.max())]})
    res.notes.append("temperature [C] from the closed-form Carslaw & Jaeger slab solution, sampled at the vertices "
                     "along the part's own axis; t = 0 is the initial state")
    return res


# -------------------------------------------------------------------------------------------- cantilever bending
def cantilever_stress_job(req: JobRequest) -> JobResult:
    """Bending stress along a cantilevered part loaded at its free tip.

    Parameters (all optional, SI): ``width``/``thickness`` [m] of the rectangular section, ``load`` [N] at the tip,
    ``length`` [m], ``axis``, ``load_steps`` (one load per scene step, default a linear ramp 0 -> ``load``),
    ``yield_mpa`` (adds the yield decision), ``fatigue_strength_coeff_pa`` + ``fatigue_strength_exponent`` (Basquin,
    adds ``fatigue_life_cycles`` and, with ``target_cycles``, the life decision).
    """
    from pinneapple_physics.closed_form.cantilever_fatigue import (
        basquin_fatigue_life_cycles,
        cantilever_bending_stress_at_position_pa,
    )

    p = dict(req.params)
    width = float(p.get("width", 0.04))
    thickness = float(p.get("thickness", 0.006))
    load = float(p.get("load", 25.0))
    u, L = local_coordinate(req.vertices, str(p.get("axis", "auto")), p.get("length"), bool(p.get("flip", False)))
    times = req.t
    steps = len(times)
    loads = np.asarray(p.get("load_steps", np.linspace(0.0, load, steps)), float)
    if loads.size != steps:
        raise ValueError(f"load_steps has {loads.size} values, the scene has {steps} time steps")

    sigma = np.asarray([[cantilever_bending_stress_at_position_pa(width, thickness, float(f), L, float(x))
                         for x in u] for f in loads], float)
    res = JobResult(req.part, req.job, {"bending_stress": (np.asarray(sigma / 1e6, dtype=np.float32), "MPa")})

    # 1. statics recomputed from first principles: M = F L at the root, W = w t^2 / 6, sigma = M / W.
    root = float(sigma[-1].max())
    mine = loads[-1] * L * 6.0 / (width * thickness ** 2)
    err = abs(root - mine) / max(abs(mine), 1e-12)
    res.checks.append(_check("root stress equals the statics value F L / W", err <= 1e-9, "reference", err, 1e-9,
                             f"{root / 1e6:.6g} MPa vs {mine / 1e6:.6g} MPa (W = w t^2 / 6 = {width * thickness ** 2 / 6:.4g} m^3)"))

    # 2. the moment line: sigma(x) / sigma(0) = 1 - x/L.
    line = 1.0 - u / L
    dev = _range_relative(float(np.abs(sigma[-1] / max(root, 1e-12) - line).max()), sigma)
    res.checks.append(_check("stress follows the moment line 1 - x/L", dev <= _REL, "physics", dev, _REL))

    # 3. free tip: no stress where there is no moment.
    tip = float(np.abs(sigma[-1]).min()) / max(root, 1e-12)
    res.checks.append(_check("free tip carries no bending stress", tip <= 1e-6, "physics", tip, 1e-6))

    if "yield_mpa" in p:
        yield_pa = float(p["yield_mpa"]) * 1e6
        peak = float(sigma.max())
        res.checks.append(_check("peak stress below yield", peak <= yield_pa, "design", peak / 1e6,
                                 float(p["yield_mpa"]),
                                 f"peak {peak / 1e6:.4g} MPa vs yield {float(p['yield_mpa']):.4g} MPa"))
    if "fatigue_strength_coeff_pa" in p:
        coeff = float(p["fatigue_strength_coeff_pa"])
        expo = float(p.get("fatigue_strength_exponent", -0.1))
        amp = 0.5 * float(sigma.max())                     # fully reversed at the mean stress of the ramp
        life = basquin_fatigue_life_cycles(amp, coeff, expo)
        finite = bool(np.isfinite(life) and life > 0)
        res.fields["fatigue_life_cycles"] = (np.full((steps, len(u)), life, dtype=np.float32), "cycles")
        if "target_cycles" in p:
            target = float(p["target_cycles"])
            res.checks.append(_check("fatigue life above the target", finite and life >= target, "design", life, target,
                                     f"Basquin at sigma_a = {amp / 1e6:.4g} MPa: N = {life:.4g} cycles, "
                                     f"target {target:.4g}"))
        else:
            res.checks.append(_check("fatigue life is finite", finite, "design", life, None,
                                     f"Basquin at sigma_a = {amp / 1e6:.4g} MPa: N = {life:.4g} cycles"))
    res.detail.update({"length_m": L, "width_m": width, "thickness_m": thickness,
                       "section_modulus_m3": width * thickness ** 2 / 6.0,
                       "load_n": [float(loads[0]), float(loads[-1])], "peak_stress_pa": float(sigma.max())})
    res.notes.append("bending stress [MPa] of a rectangular cantilever, M(x) = F (L - x), sampled at the vertices "
                     "along the part's own axis")
    return res


# ------------------------------------------------------------------------------------------------------- fin fin
def fin_temperature_job(req: JobRequest) -> JobResult:
    """Excess temperature along a convecting pin fin (the base at ``t_base``, the air at ``t_inf``).

    Parameters (SI, all from the fin's own design): ``h`` [W/m^2K], ``k`` [W/mK], ``D`` [m] diameter,
    ``L`` [m] length (default: the part's span), ``t_base``/``t_inf`` [C], ``axis``.
    The field is ``T(xi) = t_inf + (t_base - t_inf) theta(xi)`` with theta from Incropera Table 3.4 case A (convective
    tip); the check integrates the convection along the surface and the tip and compares it with the closed-form heat
    the fin dissipates -- two routes to the same number.
    """
    import math

    from pinneapple_physics.closed_form.heat import pin_fin_heat, pin_fin_theta

    p = dict(req.params)
    h = float(p.get("h", 25.0))
    k = float(p.get("k", 200.0))
    D = float(p.get("D", 0.006))
    t_base = float(p.get("t_base", 100.0))
    t_inf = float(p.get("t_inf", 25.0))
    u, L = local_coordinate(req.vertices, str(p.get("axis", "auto")), p.get("length"), bool(p.get("flip", False)))
    xi = np.clip(u / L, 0.0, 1.0)
    theta = np.asarray(pin_fin_theta(xi, h, k, D, L), float)
    temperature = t_inf + (t_base - t_inf) * theta
    steps = len(req.t)
    values = np.repeat(temperature[None, :], steps, axis=0)
    res = JobResult(req.part, req.job, {"temperature": (np.asarray(values, dtype=np.float32), "C")})

    # 1. the base is the base temperature, the tip is not (unless the fin is infinitely long)
    base_err = abs(float(theta[int(np.argmin(xi))]) - 1.0)
    res.checks.append(_check("fin base at the base temperature", base_err <= _REL, "physics", base_err, _REL,
                             f"theta at the base vertex is {1.0 - base_err:.6g}"))
    res.checks.append(_check("tip cooler than the base", float(theta.min()) < 1.0, "physics", float(theta.min()), 1.0,
                             f"theta at xi=1 is {float(theta[np.argmax(xi)]):.4g} (a fin with no convection would be 1)"))

    # 2. energy: the convection integrated along the surface plus the tip equals the closed-form heat dissipated
    xs = np.linspace(0.0, L, 401)
    th = np.asarray(pin_fin_theta(xs / L, h, k, D, L), float)
    q_integral = (t_base - t_inf) * (h * math.pi * D * float(np.trapezoid(th, xs)) + h * math.pi * D ** 2 / 4 * th[-1])
    q_closed = pin_fin_heat(h, k, D, L, t_base, t_inf)
    rel = abs(q_integral - q_closed) / max(abs(q_closed), 1e-12)
    res.checks.append(_check("dissipated heat: integral of convection equals the closed form", rel <= 1e-3, "physics",
                             rel, 1e-3, f"{q_integral:.6g} W integrated vs {q_closed:.6g} W closed form"))

    # 3. the field stays inside the two temperatures it connects
    lo, hi = min(t_base, t_inf), max(t_base, t_inf)
    over = _range_relative(max(float(lo - values.min()), float(values.max() - hi), 0.0), values)
    res.checks.append(_check("temperature stays between the air and the base", over <= _REL, "physics", over, _REL,
                             f"range [{values.min():.4g}, {values.max():.4g}] C"))
    res.detail.update({"length_m": L, "h_w_m2k": h, "k_w_mk": k, "diameter_m": D, "fin_parameter_m": math.sqrt(h * math.pi * D / (k * math.pi * D ** 2 / 4)),
                       "heat_w": float(q_closed)})
    res.notes.append("temperature [C] from the closed-form pin-fin profile (Incropera Table 3.4, case A), sampled at "
                     "the vertices along the part's own axis")
    return res


# ----------------------------------------------------------------------------------------------------- registry
JOBS: dict[str, tuple[Callable[[JobRequest], JobResult], str]] = {
    "slab_transient": (slab_transient_job,
                       "transient 1-D conduction along a part -> temperature [C], checked for residual, walls, "
                       "maximum principle and the steady limit"),
    "cantilever_stress": (cantilever_stress_job,
                          "bending stress along a cantilever loaded at the tip -> bending_stress [MPa], checked "
                          "against statics, the moment line and the yield / fatigue decisions"),
    "fin_temperature": (fin_temperature_job,
                        "excess temperature along a convecting pin fin -> temperature [C], checked at the base and "
                        "the tip and against the closed-form heat the fin dissipates"),
}


def register_job(name: str, fn: Callable[[JobRequest], JobResult], doc: str = "") -> None:
    """Make ``fn`` available to the bridge, the ingest pipeline and a ``pinneapple_link.json`` under ``name``."""
    if not callable(fn):
        raise TypeError("a job is a callable JobRequest -> JobResult")
    first = ((fn.__doc__ or "").strip().splitlines() or [""])[0]
    JOBS[str(name)] = (fn, doc or first)


def unregister_job(name: str) -> None:
    JOBS.pop(str(name), None)


def list_jobs() -> dict[str, str]:
    """The registered jobs and what each computes."""
    return {name: doc for name, (_, doc) in JOBS.items()}


def run_job(req: JobRequest) -> JobResult:
    """Run the registered job ``req.job`` on ``req``."""
    if req.job not in JOBS:
        raise KeyError(f"unknown job {req.job!r} (registered: {sorted(JOBS)})")
    fn = JOBS[req.job][0]
    out = fn(req)
    if not isinstance(out, JobResult):
        raise TypeError(f"job {req.job!r} returned {type(out).__name__}, expected JobResult")
    return out
