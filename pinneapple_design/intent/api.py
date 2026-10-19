"""``design(text)``: from a sentence to a verified design package, a list of questions, or a refusal."""
from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import pipeline as P
from .package import DesignPackage, make_geometry
from .parse import IntentLLM, OutOfScope, formulate, get_template
from .spec import RequirementSpec
from .template import CLOSED_FORM, FULL, REDUCED


@dataclass
class NeedsInput:
    """The request is in scope but a critical value is missing: answer ``questions`` and call again with
    ``answers={slot: value or (value, unit)}``."""
    spec: RequirementSpec

    @property
    def questions(self):
        return self.spec.questions

    def to_dict(self) -> dict[str, Any]:
        return {"status": "needs_input", "template": self.spec.template,
                "questions": [q.to_dict() for q in self.spec.questions], "issues": self.spec.issues,
                "requirements": self.spec.to_dict()}

    def __str__(self) -> str:
        qs = "; ".join(f"{q.text} [{q.unit}]" if q.unit else q.text for q in self.spec.questions)
        return f"Needs input ({self.spec.template}): {qs or '; '.join(self.spec.issues)}"


def design(text: str, *, answers: dict[str, Any] | None = None, llm: IntentLLM | None = None,
           template: str | None = None, seed: int = 0, out_dir: str | None = None, verify: bool = True,
           n_screen: int = 2000, budget: int = 600, n_candidates: int = 3, robust_samples: int = 2000,
           progress: Callable[[str], None] | None = None) -> DesignPackage | NeedsInput | OutOfScope:
    """Design from a plain-language goal.

    Returns a :class:`DesignPackage` (``.write(folder)`` for the files), a :class:`NeedsInput` with the questions to
    answer, or an :class:`OutOfScope` saying what would be needed. ``verify=False`` skips the full-fidelity solver
    (the trust report then says it was not run). The same ``seed`` gives the same design.
    """
    say = progress or (lambda m: None)
    spec = formulate(text, llm=llm, answers=answers, template=template)
    if isinstance(spec, OutOfScope):
        return spec
    if spec.questions or spec.issues:
        return NeedsInput(spec)
    t = get_template(spec.template)
    plan = P.plan(spec, t)
    say(f"template {t.name}: {len(plan.variables)} design variables, objectives "
        f"{[o['name'] for o in plan.objectives]}")
    log = P.CallLog()
    options: dict[str, Any] = dict(t.design_options(spec))
    rounds: list[dict[str, Any]] = []
    for round_ in range(3):
        opt = P.optimise(spec, t, seed=seed, n_screen=n_screen, budget=budget, n_candidates=n_candidates,
                         progress=say, options=options, log=log)
        if not opt.candidates:
            raise RuntimeError("the optimiser returned no candidate")
        robs = [P.robustness(spec, t, c, n=robust_samples, seed=seed, log=log, options=options,
                             level=t.robust_level, calibration=_calibration(t, spec, c, options=options))
                for c in opt.candidates]
        ok = [i for i, r in enumerate(robs) if r.ci[0] >= t.robust_target]
        best = ok[0] if ok else int(np.argmax([r.p_success for r in robs]))
        cand, rob = opt.candidates[best], robs[best]
        say(f"chosen candidate {best}: P = {rob.p_success:.3f} (95 % CI {rob.ci[0]:.3f}-{rob.ci[1]:.3f})")
        work = os.path.join(out_dir, "solver", f"round{round_}") if out_dir else None
        if not verify:
            fid = t.fidelity(FULL)
            ver = P.Verification(False, fid.label if fid else "", "skipped (verify=False)")
            break
        ver = P.verify(spec, t, cand, log=log, workdir=work, options=options)
        say("full solver: " + ("ran" if ver.ran else f"not run ({ver.reason})"))
        if not ver.ran:
            break
        new = t.calibrate(ver.result)
        full_bad = [k for k, v in ver.result.outputs.items() if k.startswith("margin") and v < 0]
        rounds.append({"round": round_, "options": dict(options), "design": dict(cand.design),
                       "measured": new, "full_margins_negative": full_bad})
        if not new or all(math.isclose(new[k], options.get(k, float("nan")), rel_tol=0.02) for k in new):
            break
        options = {**options, **new}
        y = t.evaluate(REDUCED, cand.x[None, :], spec, **options)
        still_ok = all(float(np.asarray(y[f.name])[0]) >= 0 for f in t.failure_modes if f.critical)
        if still_ok and not full_bad:
            cand.outputs.update({k: float(np.asarray(v)[0]) for k, v in y.items()})
            rob = P.robustness(spec, t, cand, n=robust_samples, seed=seed, log=log, options=options,
                               level=t.robust_level, calibration=_calibration(t, spec, cand, options=options))
            say(f"calibrated with {new}: the design still meets every margin")
            break
        say(f"calibrated with {new}: re-optimising")
    if ver.ran and ver.result is not None and t.robust_level == CLOSED_FORM:
        # the Monte Carlo runs on the closed form: calibrate its margins to the full solver at the chosen design
        rob = P.robustness(spec, t, cand, n=robust_samples, seed=seed, log=log, options=options,
                           level=CLOSED_FORM, calibration=_calibration(t, spec, cand, ver.result.outputs, options))
    assessment = P.assess(spec, t, cand, rob, ver)
    full_design = t.full_design(cand.design, spec)
    ctx: dict[str, Any] = {"verification": ver, "robustness": rob, "candidate": cand, "optimisation": opt}
    window = t.operating_window(spec, cand.design, ctx)
    ctx["window"] = window
    numbers = _numbers(t, cand, ver)
    n_cheap = log.calls[CLOSED_FORM] + log.calls[REDUCED]
    full_s = log.per_call(FULL)
    savings = {"designs_ranked_without_full_solver": int(n_cheap), "full_solver_calls": int(log.calls[FULL]),
               "full_solver_s_per_call": None if math.isnan(full_s) else full_s}
    if log.calls[FULL] and not math.isnan(full_s):
        savings["text"] = (f"{n_cheap} design evaluations ran on the closed form and the reduced model; the full "
                           f"solver ran {log.calls[FULL]} time(s) at {full_s:.1f} s each (measured). Running every "
                           f"evaluation on the full solver would have taken about {n_cheap * full_s / 3600:.1f} h "
                           "(estimate: evaluations x measured time per full call).")
    else:
        savings["text"] = (f"{n_cheap} design evaluations ran on the closed form and the reduced model; the full "
                           "solver did not run.")
    alternatives = [{"rank": c.rank, "design": c.design, "p_success": r.p_success, "ci95": list(r.ci),
                     "objectives": {o["name"]: c.outputs.get(o["name"]) for o in plan.objectives}}
                    for c, r in zip(opt.candidates, robs, strict=True)]
    geom = None
    bodies = t.geometry(spec, cand.design)
    cad = ver.result.artifacts.get("cad", {}) if ver.ran and ver.result is not None else {}
    if cad.get("stl") and len(bodies) == 1:
        # the solver's own CAD (exact holes and fillets) replaces the template's simplified drawing
        from pinneapple_design.geometry_library.core import Body
        from pinneapple_design.geometry_library.exporters import read_stl
        from pinneapple_design.geometry_library.meshops import weld
        tri = read_stl(cad["stl"]).astype(float).reshape(-1, 3)
        V, F = weld(tri, np.arange(len(tri)).reshape(-1, 3), tol=1e-9)
        bodies = [Body(bodies[0].name, V, F, bodies[0].materials)]
    if bodies:
        geom = make_geometry(t.name, t.title, bodies)
    pkg = DesignPackage(spec, plan, opt, cand, full_design, rob, alternatives, ver, assessment, window,
                        t.bill_of_materials(spec, cand.design), t.build_instructions(spec, cand.design, ctx),
                        t.limits(spec), numbers, log, savings, geom, seed=seed, rounds=rounds)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if cad.get("step"):
            import shutil
            os.makedirs(os.path.join(out_dir, "geometry"), exist_ok=True)
            pkg.files["step_exact"] = shutil.copy(cad["step"], os.path.join(out_dir, "geometry", f"{t.name}.step"))
        try:
            pkg.figures.update(t.figures(spec, cand.design, ctx, os.path.join(out_dir, "figures")))
        except ImportError as e:
            say(f"figures skipped: {e}")
        pkg.write(out_dir)
    return pkg


def _calibration(t, spec, cand: P.Candidate, full: dict[str, float] | None = None,
                 options: dict[str, Any] | None = None) -> dict[str, float]:
    """Factors on (1 + margin) that map the closed form onto the best model run on this design (the full solver's
    margins when given, else the reduced model's), both at the nominal inputs, for a robustness study on the closed
    form. ``options``: calibration options already measured (e.g. a stress-concentration factor)."""
    if t.robust_level != CLOSED_FORM:
        return {}
    keep = {k: v for k, v in (options or {}).items() if k not in t.design_options(spec)}
    x = cand.x[None, :]
    cf = t.evaluate(CLOSED_FORM, x, spec, **keep)
    best = full if full is not None else {k: float(np.asarray(v)[0]) for k, v in
                                          t.evaluate(REDUCED, x, spec, **keep).items()}
    out = {}
    for f in t.failure_modes:
        c, b = float(np.asarray(cf[f.name])[0]), best.get(f.name)
        if b is not None and 1 + c > 0 and math.isfinite(b):
            out[f.name] = (1 + b) / (1 + c)
    return out


def _numbers(t, cand: P.Candidate, ver: P.Verification) -> list[dict[str, Any]]:
    labels = {f.name: f.label for f in t.failure_modes}
    units = getattr(t, "output_units", {})
    notes = getattr(t, "output_notes", {})
    red = t.fidelity(REDUCED).label if t.fidelity(REDUCED) else REDUCED
    cf = t.fidelity(CLOSED_FORM).label if t.fidelity(CLOSED_FORM) else CLOSED_FORM
    rows = []
    for k, v in cand.outputs.items():
        rows.append({"quantity": labels.get(k, k), "key": k, "value": v, "unit": units.get(k, ""), "fidelity": red,
                     "note": "margin (>= 0 needed)" if k in labels else notes.get(k, "")})
    for k, v in cand.closed_form.items():
        if k in labels or k in ("v_top", "g_peak"):
            rows.append({"quantity": labels.get(k, k), "key": k, "value": v, "unit": units.get(k, ""),
                         "fidelity": cf, "note": "screening estimate"})
    if ver.ran and ver.result is not None:
        for k, v in ver.result.outputs.items():
            rows.append({"quantity": k, "key": k, "value": v, "unit": units.get(k, ""), "fidelity": ver.label,
                         "note": "full solver"})
    return rows
