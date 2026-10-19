"""Identify the governing equations of a problem from its description and, when given, from data.

:func:`identify` returns ranked candidates, each a complete equation system (PDEs, conservation law, constitutive
relation, dimension, conditions, coefficients) with a confidence, the reasoning and decision records with their
references. The heuristic path is offline (keywords, units, explicit values; :func:`pinneapple_physics.pde_environment.
capabilities.identify_pde` as a second opinion); an LLM, when configured, adds a candidate that is parsed and checked
like any other. :func:`discover_pde` fits the PDE to gridded space-time data by sparse regression over candidate
derivative terms (STRidge, :class:`pinneapple_analysis.inverse_problems.missing_term.SINDyIdentifier`) and says whether
the data confirm a candidate.

    res = identify("a steel rod cools down, its ends held at 20 C, diffusivity 1.2e-5 m^2/s")
    best = res["candidates"][0]           # family, pdes, conditions, coefficients, confidence, reasoning, decisions
    found = discover_pde(axes, u, ["x", "t"])   # {"equation": "u_t = 0.0998*u_xx", "r2": ..., "family": "heat"}
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .decisions import llm_decision, rule_decision
from .planner.heuristic import _equation_params, detect
from .templates import DESCRIPTIONS, FAMILIES

PHYSICS = {
    "heat": ("energy conservation", "Fourier's law q = -k grad(T)"),
    "wave": ("momentum balance (Newton's second law on the string / membrane / medium)", "linear restoring force "
             "(tension or elasticity)"),
    "poisson": ("steady conservation with a source", "flux proportional to the gradient (Fourier / Gauss / Darcy)"),
    "laplace": ("steady conservation without sources", "flux proportional to the gradient"),
    "helmholtz": ("time-harmonic form of the wave equation", "linear medium"),
    "burgers": ("momentum conservation in 1-D", "Newtonian viscosity"),
    "advection_diffusion": ("mass conservation of a transported quantity", "Fick's law plus advection by a "
                            "given velocity"),
    "reaction_diffusion": ("mass conservation with a source", "Fick's law plus logistic growth"),
}


def _capabilities_second_opinion(text: str) -> dict[str, float]:
    try:
        from pinneapple_physics.pde_environment.capabilities import identify_pde
        return {k: float(s) for k, s in identify_pde(text)}
    except Exception:  # noqa: BLE001 - a second opinion only
        return {}


def _candidate(det: dict[str, Any], family: str, dim: int, share: float, why: list[str]) -> dict[str, Any]:
    eqp, t = _equation_params(det, family, dim)
    conservation, constitutive = PHYSICS.get(family, ("", ""))
    decs = [rule_decision(f"family.{family}" if f"family.{family}" in _rules() else "family.keywords",
                          what=f"{family}: {'; '.join(eqp['pdes'])}", node="problem", value=eqp["pdes"],
                          rationale="; ".join(why)).to_dict()]
    decs.append(rule_decision("dimension.keywords" if det.get("dimension") else "dimension.default",
                              what=f"{dim}-D", node="problem", value=dim).to_dict())
    decs.append(rule_decision("bc.insulated" if "neumann" in det.get("boundary", []) else "bc.dirichlet_template",
                              what="boundary and initial conditions", node="problem",
                              value=eqp["conditions"]).to_dict())
    for k, v in eqp["coefficients"].items():
        decs.append(rule_decision("coef.from_text" if k in det.get("coefficients", {}) else "coef.template_default",
                                  what=f"{k} = {v:g}", node="problem", value={k: v}).to_dict())
    conf = round(min(0.95, 0.35 + 0.6 * share), 2)
    return {"family": family, "description": DESCRIPTIONS[family], "dimension": dim,
            "time_dependent": bool(t["time"]), "fields": eqp["fields"], "pdes": eqp["pdes"],
            "conditions": eqp["conditions"], "coefficients": eqp["coefficients"], "exact": eqp["exact"],
            "bounds": t["bounds"], "time": t["time"], "conservation_law": conservation,
            "constitutive_relation": constitutive, "confidence": conf, "reasoning": why, "source": "heuristic",
            "decisions": decs}


def _rules():
    from .rules import RULES
    return RULES


def identify(text: str, *, dimension: int | None = None, units: dict[str, str] | None = None, n: int = 3,
             llm: dict[str, str] | None = None, call=None) -> dict[str, Any]:
    """Ranked candidate equation systems for a description (see the module text). ``dimension`` overrides the
    detected one; ``units`` (quantity -> unit) are recorded; ``llm`` = ``{"provider", "model", "base_url",
    "api_key"}`` adds an LLM candidate (validated; dropped with a warning when it does not parse)."""
    det = detect(text or "")
    scores = dict(det.get("family_scores") or {})
    second = _capabilities_second_opinion(text or "")
    for fam, s in second.items():                       # the library's own family tags, at half weight
        f = {"heat_equation": "heat", "wave_equation": "wave", "poisson": "poisson", "laplace": "laplace",
             "burgers": "burgers", "helmholtz": "helmholtz", "advection_diffusion": "advection_diffusion",
             "reaction_diffusion": "reaction_diffusion"}.get(fam)
        if f:
            scores[f] = scores.get(f, 0.0) + 0.5 * s
    if det.get("steady") and "heat" in scores:          # steady heat is Laplace / Poisson
        target = det.get("family") if det.get("family") in ("laplace", "poisson") else "laplace"
        scores[target] = scores.get(target, 0.0) + scores.pop("heat")
    warnings: list[str] = []
    if not scores:
        warnings.append("no physics recognised: candidates are generic starting points with low confidence")
        scores = {"heat": 0.3, "poisson": 0.2, "wave": 0.1}
    total = sum(scores.values())
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:n]
    cands = []
    for fam, s in ranked:
        if fam not in FAMILIES:
            continue
        dim = dimension or det.get("dimension") or (1 if fam == "burgers" else 2)
        if fam == "burgers":
            dim = 1
        why = [f"keyword score {s:.2f} of {total:.2f}"]
        if second:
            why.append("pinneapple_physics capabilities.identify_pde: " + ", ".join(f"{k} {v:g}" for k, v in
                                                                                    list(second.items())[:3]))
        if det.get("steady"):
            why.append("steady wording in the text")
        if det.get("boundary"):
            why.append(f"boundary wording: {', '.join(det['boundary'])}")
        if det.get("coefficients"):
            why.append(f"values in the text: {det['coefficients']}")
        if det.get("quantities"):
            why.append("quantities: " + ", ".join(q["quote"] for q in det["quantities"][:5]))
        c = _candidate(det, fam, int(dim), s / total, why)
        if warnings:                                    # nothing recognised: a starting point, not an identification
            c["confidence"] = min(c["confidence"], 0.3)
            c["reasoning"].append("nothing in the text points to this family: a starting point to edit")
        cands.append(c)
    if llm and llm.get("provider"):
        c = _llm_candidate(text, llm, call)
        if isinstance(c, dict):
            cands.insert(0 if c["confidence"] > (cands[0]["confidence"] if cands else 0) else 1, c)
        else:
            warnings.append(c)
    return {"candidates": cands, "detected": det, "units": dict(units or {}), "warnings": warnings}


def _llm_candidate(text: str, llm: dict[str, str], call=None) -> dict[str, Any] | str:
    from pinneapple_physics.symbolic_pde.text import EquationSyntaxError, parse_system

    from .planner.llm import _extract_json, _provider_kwargs
    from .references import registry
    if call is None:
        from pinneapple_llm import call_llm as call
    keys = ", ".join(registry())
    prompt = ("Identify the governing equations of this problem. Answer one JSON object: {\"family\", "
              "\"dimension\" (1-3), \"fields\": [..], \"pdes\": [..], \"conditions\": [{\"kind\", \"on\", "
              "\"equation\"}], \"coefficients\": {..}, \"conservation_law\", \"constitutive_relation\", \"reasoning\", "
              "\"confidence\" (0-1), \"references\": [registry keys or {title, doi}]}. Grammar: u_t, u_xx, lap(u), "
              f"d(f, x); coordinates x, y, z, t. Registry keys: {keys}.\n\nProblem:\n{text}")
    prov = llm["provider"]
    try:
        raw = call(prompt, provider=prov, model=llm.get("model") or None, api_key=llm.get("api_key") or None,
                   system="You identify PDE models. JSON only. Never invent references.", json_mode=True,
                   module="pinneapple_flow.identify", max_tokens=2000, **_provider_kwargs(prov, llm.get("base_url",
                                                                                                    "")))
        d = _extract_json(raw)
        dim = int(d.get("dimension") or 1)
        coords = ["x", "y", "z"][:dim] + (["t"] if any("_t" in p or "d(" in p for p in d.get("pdes", [])) else [])
        parse_system(list(d["pdes"]), list(d.get("conditions") or []), coords=coords,
                     fields=list(d.get("fields") or ["u"]), coefficients=dict(d.get("coefficients") or {}))
    except (EquationSyntaxError, KeyError, ValueError, TypeError) as exc:
        return f"LLM candidate rejected: {exc}"
    except Exception as exc:  # noqa: BLE001 - provider failure
        return f"LLM unavailable: {type(exc).__name__}: {exc}"
    who = f"{prov}/{llm.get('model') or 'default'}"
    dec = llm_decision(who, topic="equations", what=f"{d.get('family')}: {'; '.join(d['pdes'])}", node="problem",
                       value=d["pdes"], rationale=str(d.get("reasoning", ""))[:600],
                       references=list(d.get("references") or [])[:8]).to_dict()
    conf = float(d.get("confidence") or 0.5)
    conf = round(min(conf, dec["confidence"]) if dec["no_reference"] else min(conf, 0.9), 2)
    return {"family": str(d.get("family") or "custom"), "description": str(d.get("reasoning", ""))[:300],
            "dimension": dim, "time_dependent": "t" in coords, "fields": list(d.get("fields") or ["u"]),
            "pdes": list(d["pdes"]), "conditions": list(d.get("conditions") or []),
            "coefficients": dict(d.get("coefficients") or {}), "exact": "",
            "bounds": {c: [0.0, 1.0] for c in coords if c != "t"}, "time": [0.0, 1.0] if "t" in coords else None,
            "conservation_law": str(d.get("conservation_law", "")), "constitutive_relation":
                str(d.get("constitutive_relation", "")), "confidence": conf,
            "reasoning": [str(d.get("reasoning", ""))], "source": f"llm:{who}", "decisions": [dec]}


# ------------------------------------------------------------------------------------------------ discovery
def _fd_derivatives(U: np.ndarray, axes: dict[str, np.ndarray], coords: list[str]):
    d = {}
    for i, c in enumerate(coords):
        h = float(axes[c][1] - axes[c][0])
        g1 = np.gradient(U, h, axis=i, edge_order=2)
        d[c] = g1
        d[c + c] = np.gradient(g1, h, axis=i, edge_order=2)
    return d


def discover_pde(axes: dict[str, np.ndarray], values: np.ndarray, coords: list[str], *, field: str = "u",
                 threshold: float = 0.05, trim: int = 3) -> dict[str, Any]:
    """Sparse regression of ``u_t`` (and ``u_tt``) on candidate terms (u, u^2, u^3, first and second spatial
    derivatives, u u_x) of gridded data ``values`` (shape: axis lengths in ``coords`` order). Returns the equation
    with the best fit, its coefficients, R^2 and the family it classifies as."""
    from pinneapple_analysis.inverse_problems.missing_term import SINDyIdentifier
    from pinneapple_physics.symbolic_pde.text import parse_system
    if "t" not in coords:
        raise ValueError("equation discovery here needs time-dependent data (a 't' axis)")
    U = np.asarray(values, float)
    if U.ndim == len(coords) + 1:
        U = U[..., 0]
    der = _fd_derivatives(U, axes, coords)
    spatial = [c for c in coords if c != "t"]
    terms = {field: U, f"{field}**2": U ** 2, f"{field}**3": U ** 3}
    for c in spatial:
        terms[f"{field}_{c}"] = der[c]
        terms[f"{field}_{c}{c}"] = der[c + c]
        terms[f"{field}*{field}_{c}"] = U * der[c]
    sl = tuple(slice(trim, -trim) if n > 2 * trim + 2 else slice(None) for n in U.shape)
    names = list(terms)
    Theta = np.stack([terms[k][sl].ravel() for k in names], 1)
    Z = Theta - Theta.mean(0)
    Z /= np.maximum(np.linalg.norm(Z, axis=0), 1e-300)
    C = np.abs(Z.T @ Z)
    collinear = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names)) if C[i, j] > 0.999]
    best = None
    for target, tname in ((der["t"], f"{field}_t"), (der["tt"], f"{field}_tt")):
        b = target[sl].ravel()
        scale = float(np.linalg.norm(b)) or 1.0
        res = SINDyIdentifier(threshold=threshold, method="stridge").fit(Theta, b / scale, names)
        xi = res.coefficients * scale
        pred = Theta @ xi
        r2 = 1.0 - float(np.sum((b - pred) ** 2) / max(np.sum((b - b.mean()) ** 2), 1e-300))
        if best is None or r2 > best["r2"] + 1e-3:
            best = {"lhs": tname, "terms": {n: float(c) for n, c in zip(names, xi, strict=True) if c != 0.0},
                    "r2": r2}
    rhs = " + ".join(f"({c:.6g})*{n}" for n, c in best["terms"].items()) or "0"
    eq = f"{best['lhs']} = {rhs}"
    sys_ = parse_system([eq], [{"kind": "initial", "equation": f"{field} = 0"}], coords=coords, fields=[field])
    fam = sys_.classify()["family"]
    pretty = f"{best['lhs']} = " + " + ".join(f"{c:.4g}*{n}" for n, c in best["terms"].items())
    dec = rule_decision("discovery.sparse_regression", what=f"discovered: {pretty}", node="problem",
                        value=best["terms"], rationale=f"STRidge on {len(names)} candidate terms, R^2 = "
                                                       f"{best['r2']:.4f} on {Theta.shape[0]} interior points")
    active = set(best["terms"])
    ambiguous = [p for p in collinear if p[0] in active or p[1] in active]
    return {"equation": eq, "pretty": pretty, "terms": best["terms"], "r2": best["r2"], "family": fam,
            "n_points": int(Theta.shape[0]), "candidates": names, "decision": dec.to_dict(),
            "collinear": ambiguous,
            "warning": (f"terms {ambiguous} are nearly collinear in these data (e.g. a single spatial mode): the data "
                        "cannot tell them apart; use data with richer content") if ambiguous else ""}


def confirm(candidate: dict[str, Any], found: dict[str, Any]) -> dict[str, Any]:
    """Does the discovered equation agree with a candidate (same family; coefficients where they can be read)?"""
    out = {"family_agrees": found["family"] == candidate["family"], "coefficients": {}}
    lap = [v for k, v in found["terms"].items() if k.endswith(("_xx", "_yy", "_zz"))]
    for k in ("alpha", "nu", "D"):
        if k in candidate["coefficients"] and lap:
            out["coefficients"][k] = {"text": candidate["coefficients"][k], "data": float(np.mean(lap))}
    if "c" in candidate["coefficients"] and found["lhs" if "lhs" in found else "equation"]:
        if lap and found["equation"].startswith(("u_tt",)):
            out["coefficients"]["c"] = {"text": candidate["coefficients"]["c"], "data": float(np.sqrt(abs(np.mean(lap))))}
    return out


def to_json(res: dict[str, Any]) -> str:
    return json.dumps(res, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
