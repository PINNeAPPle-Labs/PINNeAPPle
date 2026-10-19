"""Validation nodes: compare solutions with a reference (exact, benchmark or another method), trust card, and the
PINNeAPPle Lab curation (seal letters A-F) with the quality seal on the figure."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ._common import exact_function, grid_points, interpolate, load_solution, need_problem


def _pointwise(fn, coords):
    """A compiled expression as a function of points ``P[:, k]`` (one column per coordinate, in order)."""
    def at(P):
        return np.broadcast_to(np.asarray(fn({c: P[:, k] for k, c in enumerate(coords)}), float), (len(P),))
    return at


def _trust_card(problem, ref_sol, cand_vals, ref_vals):
    """``pp.trust_card`` on the candidate's grid for steady 2-D Laplace / Poisson (residual, conservation, boundary
    values) and the reference; None when the problem is outside what the card checks."""
    import pinneapple as pp
    eq = problem["equations"]
    fam = (eq.get("classification") or {}).get("family") or eq.get("family")
    coords = ref_sol["coords"]
    if fam not in ("laplace", "poisson") or coords != ["x", "y"]:
        return None
    grid = {c: ref_sol["axes"][c] for c in coords}
    kw = {}
    if fam == "poisson":
        from pinneapple_physics.symbolic_pde.text import compile_expr

        from ._common import system_of
        sys_ = system_of(problem)
        import sympy as sp
        u = sys_.ctx.functions[sys_.fields[0]]
        R = sys_.substituted()[0].doit()
        lap = sum(sp.Derivative(u, sys_.ctx.symbols[c], 2) for c in coords)
        rest = sp.expand(R - lap)
        if rest.has(u):
            return None
        f = compile_expr((-rest).xreplace({s: sp.Symbol(s.name) for s in rest.free_symbols}), "numpy")
        kw["source"] = _pointwise(f, coords)
    from pinneapple_physics.symbolic_pde.text import compile_expr

    from ._common import system_of
    sys_ = system_of(problem)
    dirichlet = {}
    for c in sys_.conditions:
        if c.kind != "dirichlet" or c.lhs != sys_.ctx.functions[sys_.fields[0]]:
            continue
        import sympy as sp
        g = compile_expr(c.rhs.xreplace({s: sp.Symbol(s.name) for s in c.rhs.free_symbols}), "numpy")
        faces = [f"{a}_{e}" for a in coords for e in ("min", "max")] if c.on == "boundary" else [c.on]
        for face in faces:
            dirichlet[face] = _pointwise(g, coords)
    if dirichlet:
        kw["dirichlet"] = dirichlet
    return pp.trust_card(grid, cand_vals, pde=fam, reference=ref_vals, name="field", is_ml=True, **kw)


@node("validate.compare", "Validation vs reference", category="Validation",
      inputs=[Port("candidates", "solution", multiple=True), Port("reference", "solution", required=False),
              Port("problem", "problem", required=False)],
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("reference", "enum", "auto", choices=["auto", "exact", "input"],
                    help="auto: the exact solution when the problem has one, else the reference input"),
              Param("tolerance", "float", 0.02, minimum=0.0, help="relative L2 that passes"),
              Param("trust_card", "bool", True, help="pp.trust_card where it applies (steady 2-D Laplace/Poisson)")],
      calls="relative L2 / max / RMSE on common points + pp.trust_card")
def compare(ctx, inputs, params):
    """Measure every candidate against the reference on the same points (relative L2, max and RMS error) and, where
    it applies, the trust card's physics checks; one row per candidate."""
    cands = [load_solution(v) for v in inputs["candidates"]]
    if not cands:
        raise StepFailed("connect at least one candidate solution")
    problem = need_problem(inputs["problem"]) if inputs.get("problem") else None
    mode = params["reference"]
    exact = exact_function(problem) if problem else None
    ref = load_solution(inputs["reference"]) if inputs.get("reference") else None
    if mode == "exact" and exact is None:
        raise StepFailed("reference 'exact' needs a problem with an exact solution")
    if mode == "input" and ref is None:
        raise StepFailed("reference 'input' needs a reference solution")
    use_exact = exact is not None and (mode == "exact" or (mode == "auto"))
    if not use_exact and ref is None:
        raise StepFailed("no reference: connect a reference solution or a problem with an exact solution")
    base = cands[0]
    ctx.decide("validation.reference", f"reference: {'exact solution' if use_exact else 'reference input'}",
               value=mode)
    ctx.auto("validation.threshold", f"pass at relative L2 <= {params['tolerance']:g}", "tolerance",
             value=params["tolerance"])
    if "tolerance" in ctx.explicit and params["tolerance"] != 0.02:
        ctx.user_choice("validation", f"pass threshold set to {params['tolerance']:g}", params["tolerance"])
    X = grid_points(base["axes"], base["coords"])
    if use_exact:
        R = exact(X)
        ref_name = "exact solution"
    else:
        R = interpolate(ref, X)
        ref_name = f"{ref['label']} ({ref['method']})"
    rows, checks, results = [], [], {}
    for s in cands:
        V = s["values"].reshape(-1, s["values"].shape[-1]) if s["coords"] == base["coords"] and s["values"].shape == \
            base["values"].shape else interpolate(s, X)
        err = V[:, :R.shape[1]] - R
        rel = float(np.linalg.norm(err) / max(np.linalg.norm(R), 1e-300))
        mx = float(np.abs(err).max())
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ok = rel <= float(params["tolerance"])
        results[s["label"]] = {"rel_l2": rel, "max_abs": mx, "rmse": rmse, "passed": ok, "method": s["method"]}
        rows.append([s["label"], s["method"], rel, mx, rmse, "PASS" if ok else "FAIL"])
        checks.append({"name": f"{s['label']}_rel_l2_vs_{'exact' if use_exact else 'reference'}", "value": rel,
                       "max": float(params["tolerance"]), "passed": ok, "kind": "reference" if use_exact else
                       "verification", "detail": f"relative L2 against the {ref_name} on {len(X)} points"})
        ctx.metric(f"{s['label']}_rel_l2", rel, f"against the {ref_name} on {len(X)} points")
        if params["trust_card"] and problem is not None and s["coords"] == ["x", "y"]:
            # on the candidate's own grid: interpolating it onto another grid would change its derivatives
            Xs = grid_points(s["axes"], s["coords"])
            Rs = exact(Xs) if use_exact else interpolate(ref, Xs)
            shp = tuple(len(s["axes"][c]) for c in s["coords"])
            card = _trust_card(problem, s, s["values"][..., 0].reshape(shp), Rs[:, 0].reshape(shp))
            if card is not None:
                ctx.table(f"trust card: {s['label']} ({card.decision})", ["check", "passed", "value", "detail"],
                          [[c.name, c.passed, c.value, c.detail] for c in card.checks])
                results[s["label"]]["trust_card"] = card.decision
                checks.append({"name": f"{s['label']}_trust_card", "passed": card.decision == "APPROVED",
                               "kind": "physics", "value": card.decision,
                               "detail": "pp.trust_card: residual, conservation, boundary, reference"})
    ctx.table(f"errors against the {ref_name}", ["candidate", "method", "rel. L2", "max |err|", "RMSE", "verdict"],
              rows)
    own_rows = []
    for s in cands:                                  # a candidate's own checks (validate_pinn for a PINN) count too
        for c in s.get("own_checks") or []:
            checks.append({"name": f"{s['label']}_{c['id']}", "passed": bool(c["passed"]), "value": c["value"],
                           "kind": "physics", "detail": f"{s['label']}'s own check: {c['detail']}"})
            own_rows.append([s["label"], c["id"], "PASS" if c["passed"] else "FAIL", c["detail"]])
        failed = [c for c in s.get("own_checks") or [] if not c["passed"]]
        if failed and results.get(s["label"], {}).get("passed"):
            msg = (f"{s['label']}: {', '.join(c['id'] for c in failed)} fails ({failed[0]['detail']}) although its "
                   f"relative L2 against the {ref_name} is {results[s['label']]['rel_l2']:.2e}: a small global "
                   "error with a check that fails locally; the failing check is passed to the Lab curation")
            ctx.warn(msg)
            results[s["label"]]["own_checks_failed"] = [c["id"] for c in failed]
    if own_rows:
        ctx.table("candidates' own checks", ["candidate", "check", "verdict", "detail"], own_rows)
    if len(cands) > 1:
        best = min(results, key=lambda k: results[k]["rel_l2"])
        ctx.metric("best", best)
    fig = _error_figure(base, cands, X, R, ref_name)
    fa = ctx.figure(fig, "validation", f"candidates against the {ref_name}")
    ctx.summary(", ".join(f"{k}: {v['rel_l2']:.2e}" for k, v in results.items()) + f" (vs {ref_name})")
    return {"metrics": {"reference": ref_name, "candidates": results, "checks": checks,
                        "figure": fa["$artifact"]}, "figure": fa}


def _error_figure(base, cands, X, R, ref_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    coords = base["coords"]
    spatial = [c for c in coords if c != "t"]
    shp = tuple(len(base["axes"][c]) for c in coords)
    Rg = R[:, 0].reshape(shp)
    if "t" in coords:
        Rg = np.moveaxis(Rg, coords.index("t"), 0)[-1]
    n = len(cands)
    if len(spatial) == 1:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        x = base["axes"][spatial[0]]
        ax.plot(x, Rg, "k-", lw=2.5, label=ref_name)
        for s in cands:
            V = interpolate(s, X)[:, 0].reshape(shp)
            if "t" in coords:
                V = np.moveaxis(V, coords.index("t"), 0)[-1]
            ax.plot(x, V, "--", label=s["label"])
        ax.legend(fontsize=8)
        ax.set_xlabel(spatial[0])
        ax.set_title("final time" if "t" in coords else "solution", fontsize=9)
        return fig
    fig, axs = plt.subplots(1, n + 1, figsize=(3.6 * (n + 1), 3.3), squeeze=False, constrained_layout=True)
    axs = axs[0]
    a, b = (base["axes"][c] for c in spatial[:2])
    sl = (slice(None), slice(None)) + ((Rg.shape[2] // 2,) if Rg.ndim == 3 else ())
    im = axs[0].pcolormesh(a, b, Rg[sl].T, shading="auto")
    fig.colorbar(im, ax=axs[0])
    axs[0].set_title(ref_name, fontsize=9)
    for ax, s in zip(axs[1:], cands, strict=True):
        V = interpolate(s, X)[:, 0].reshape(shp)
        if "t" in coords:
            V = np.moveaxis(V, coords.index("t"), 0)[-1]
        im = ax.pcolormesh(a, b, (V - Rg)[sl].T, shading="auto", cmap="coolwarm")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"error: {s['label']}", fontsize=9)
    for ax in axs:
        ax.set_aspect("equal")
    return fig


@node("validate.lab", "Lab curation + seal", category="Validation",
      inputs=[Port("validation", "metrics", multiple=True)],
      outputs=[Port("report", "report"), Port("figure", "figure", required=False)],
      params=[Param("title", "str", "", help="item title in the Lab catalogue"),
              Param("baseline_check", "bool", True, help="count 'best of several methods' as a baseline check")],
      calls="pinneapple_lab.runner.run + curation.curate (letters A-F) + curation.next_steps + seal.seal_for / "
            "stamp_figure")
def lab(ctx, inputs, params):
    """Record the validation as a PINNeAPPle Lab run, grade it (letter A-F: reference, baseline, physics checks,
    reproducibility, assets, documentation) and, for tiers A and B only, stamp the quality seal on the figure."""
    from pinneapple_lab.curation import curate
    from pinneapple_lab.runner import run as lab_run
    from pinneapple_lab.seal import SealError, seal_for, stamp_file
    from pinneapple_lab.spec import Experiment
    from pinneapple_lab.store import LabStore

    from ..config import home
    vals = [v for v in inputs["validation"] if isinstance(v, dict) and "checks" in v]
    if not vals:
        raise StepFailed("connect the metrics of a validation node")
    checks = [c for v in vals for c in v["checks"]]
    cand = {k: r for v in vals for k, r in (v.get("candidates") or {}).items()}
    figure = next((v.get("figure") for v in vals if v.get("figure")), None)
    flow_id = re.sub(r"[^a-z0-9_]+", "_", str(ctx.meta.get("flow_id", "flow")).lower())
    name = f"flow_{flow_id}_{re.sub(r'[^a-z0-9_]+', '_', ctx.node_id.lower())}"[:80]
    digest = hashlib.sha1(json.dumps([checks, cand], sort_keys=True, default=str).encode()).hexdigest()[:12]
    root = str(home() / "lab")

    class FlowValidation(Experiment):
        pass
    FlowValidation.name = name
    FlowValidation.description = params["title"] or f"Flow Studio validation {ctx.node_id}"
    FlowValidation.tags = ["flow-studio"]
    FlowValidation.params = {"evidence": digest}

    def run_exp(self, lctx):
        for k, r in cand.items():
            lctx.metric(f"{k}_rel_l2", r["rel_l2"])
        for c in checks:
            lctx.check(c["name"], bool(c["passed"]), value=c.get("value") if isinstance(c.get("value"), (int, float))
                       else None, max=c.get("max"), kind=c.get("kind"), detail=c.get("detail", ""))
        if params["baseline_check"] and len(cand) > 1:
            best = min(cand, key=lambda k: cand[k]["rel_l2"])
            worst = max(cand, key=lambda k: cand[k]["rel_l2"])
            lctx.check(f"{best}_beats_{worst}", cand[best]["rel_l2"] <= cand[worst]["rel_l2"], kind="baseline",
                       detail=f"{best} {cand[best]['rel_l2']:.2e} vs {worst} {cand[worst]['rel_l2']:.2e}")
        if figure:
            import os
            import shutil
            dst = lctx.path("figures", "validation.png")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(figure, dst)
            lctx.files.setdefault("figures", []).append(os.path.relpath(dst, lctx.dir))
    FlowValidation.run = run_exp
    ctx.decide("validation.lab_tiers", "grade with the Lab curation (letters A-F)")
    res = lab_run(FlowValidation, root=root)
    cur = curate(LabStore(root))
    item = cur["items"].get(name, {})
    item_tier = item.get("tier", "?")
    run_rec = (cur.get("runs") or {}).get(res.run_id) or {}
    hl = run_rec.get("headline") or {}
    tier = run_rec.get("tier") or item_tier             # the letter the curation gives THIS run
    ctx.metric("lab_status", res.status)
    from pinneapple_lab.curation import TIERS
    ctx.metric("tier", tier, "this run, graded by pinneapple_lab.curation: " +
               ", ".join(f"{k} {v}" for k, v in TIERS.items()) + (f"; {hl['text']}" if hl.get("text") else ""))
    ctx.metric("item_tier", item_tier, "the Lab item's letter: the best of its runs")
    ctx.metric("lab_run_id", res.run_id)
    if tier != item_tier:
        ctx.warn(f"this run is graded {tier} ({run_rec.get('tier_name', '')}: {hl.get('text', '')}"
                 f"{'; ' + hl['hint'] if hl.get('hint') else ''}); the item keeps {item_tier} from its best run "
                 f"({item.get('best_run', '')})")
    from ..lab_status import next_steps, seal_status
    st = seal_status(name, root)                                 # the Lab's letter (A-F once it has that scale)
    steps = next_steps(name, root)
    ctx.report["lab"] = {"item": name, "root": root, "seal": st, "next_steps": steps}
    if st["letter"]:
        ctx.metric("seal", st["letter"], f"pinneapple_lab ({st['scale']}): {st['name']}")
    if st["failing_check"]:
        ctx.warn(f"failing check: {st['failing_check']}" + (f"; to raise it: {st['hint']}" if st["hint"] else ""))
    if steps["gaps"]:
        ctx.table("what the next letter needs", ["gap", "suggested action", "nodes"],
                  [[g["gap"], g["action"] or g["command"], ", ".join(n["type"] for n in g["nodes"])]
                   for g in steps["gaps"]])
    out = {"report": ctx.save_json("lab", {"tier": tier, "run": res.run_id, "item": item, "status": res.status,
                                           "seal": st, "next_steps": steps})}
    try:
        seal = seal_for(name, root)
    except (SealError, KeyError) as exc:
        ctx.warn(f"no seal: {exc}")
        ctx.summary(f"tier {tier}: not sealed ({exc})")
        return out
    if figure and seal.tier == tier:                 # stamp only a seal that is this run's letter
        sealed = stamp_file(figure, seal, out=str(ctx.path("validation_sealed.png")))
        from ..runtime import artifact
        out["figure"] = artifact(sealed, "png")
        ctx.report["figures"].append({"path": str(sealed), "name": "validation_sealed",
                                      "caption": f"tier {tier} seal"})
    ctx.summary(f"tier {tier}" + (f" ({hl['text']})" if hl.get("text") and tier in ("D", "E", "F") else "") +
                (", sealed" if seal.tier == tier else f"; the item's seal is {seal.tier} from an earlier run"))
    return out
