"""Trust of a chain: the checks behind the final design, as a :class:`~pinneapple_analysis.trust.TrustReport` and as
the six-stage trust card of :func:`pinneapple_lab.curation.trust_stages`.

Checks gathered (kind used by the six-stage card in brackets):

* every requirement's failure mode on the final version, critical [decision: through the report's verdict];
* inputs recorded and finite [sanity -> data and geometry];
* the known-answer cases of every failure mode used [verification -> model];
* each computation's internal checks: energy balance closure, fixed-point convergence [physics];
* solver-against-closed-form agreements (finite-difference beam, Rayleigh frequency, MOC vs Joukowsky, ...)
  [reference -> benchmark];
* the worst margin when each uncertain premise moves by its declared spread, one at a time, and the applicability
  range of each model [uncertainty].
"""
from __future__ import annotations

import math
from typing import Any

from pinneapple_analysis.trust import Check, TrustReport

from .chain import Chain, text_of
from .library import LIBRARY, iter_known_answers

__all__ = ["chain_checks", "trust_report", "trust_card", "known_answer_checks", "uncertainty_checks"]


def _kind_of_self_check(c: Check) -> str:
    return "reference" if c.name.startswith("solver vs closed form") else "physics"


def known_answer_checks(modes) -> list[Check]:
    out = []
    for m, case in iter_known_answers(sorted(set(modes))):
        v = m.evaluate(case["params"])
        got = v.quantities[case["quantity"]][0]
        rel = abs(got - case["expected"]) / abs(case["expected"])
        out.append(Check(f"known answer: {m.name}.{case['quantity']}", "supports" if rel <= case["rtol"] else
                         "contradicts", value=rel, limit=case["rtol"], source=case["source"]))
    return out


def uncertainty_checks(chain: Chain) -> list[Check]:
    """Worst margin of each final check when one uncertain premise at a time moves by +- its relative spread."""
    final = chain.final
    out = []
    for v in final.verifications:
        m = LIBRARY.mode(v.mode)
        lim = next((r.limits for r in chain.requirements if r.id == v.requirement), {})
        worst, why = v.margin, "nominal"
        for name, spread in m.uncertain.items():
            if name not in final.concept.params or not isinstance(final.concept.params[name], (int, float)):
                continue
            for s in (-1, 1):
                p = {**final.concept.params, name: final.concept.params[name] * (1 + s * spread),
                     "_elements": tuple(final.concept.elements)}
                try:
                    vv = m.evaluate(p, v.requirement, lim)
                except (ValueError, ZeroDivisionError, KeyError):
                    continue
                if vv.margin < worst:
                    worst, why = vv.margin, f"{name} {'+' if s > 0 else '-'}{100 * spread:.0f} %"
        out.append(Check(f"uncertainty: {v.requirement}/{v.mode} worst margin ({why})",
                         "supports" if worst >= 0 else "contradicts", value=worst if math.isfinite(worst) else None,
                         limit=0.0, source="one-at-a-time spread of the mode's uncertain premises"))
    return out


def chain_checks(chain: Chain) -> list[tuple[Check, str]]:
    """``(check, kind)`` for the final version of ``chain``."""
    final = chain.final
    out: list[tuple[Check, str]] = []
    params = final.concept.params
    bad = [k for k, x in params.items() if isinstance(x, float) and not math.isfinite(x)]
    out.append((Check("inputs recorded and finite", "supports" if not bad else "contradicts",
                      value=float(len(params)), source="concept parameters with their sources",
                      reason=", ".join(bad)), "sanity"))
    for v in final.verifications:
        out.append((v.check(), "decision"))
        for c in v.self_checks:
            out.append((c, _kind_of_self_check(c)))
        for val in v.validity:
            out.append((Check(f"applicability: {v.mode}: {val.name}", "supports" if val.ok else "contradicts",
                              value=val.value if math.isfinite(val.value) else None,
                              limit=val.hi if val.hi is not None else val.lo, source=val.note), "uncertainty"))
    out += [(c, "verification") for c in known_answer_checks([v.mode for v in final.verifications])]
    out += [(c, "uncertainty") for c in uncertainty_checks(chain)]
    return out


def trust_report(chain: Chain, **kw: Any) -> TrustReport:
    return TrustReport([c for c, _ in chain_checks(chain)], model=text_of(chain.title, "en"), **kw)


def trust_card(chain: Chain, out_dir: str = ".", report: TrustReport | None = None) -> dict[str, dict[str, Any]]:
    """The six questions (data and geometry, model, physics, benchmark, uncertainty, decision) answered from the
    chain's own checks with :func:`pinneapple_lab.curation.trust_stages`."""
    from pinneapple_lab.curation import trust_stages
    rep = report or trust_report(chain)
    checks = [{"name": c.name, "passed": c.status == "supports", "kind": k,
               "detail": (c.reason or c.source or "")[:120]} for c, k in chain_checks(chain)
              if c.status != "not_run" and k != "decision"]
    tier = {"APPROVED": "B", "REVIEW": "C", "REJECT": "E"}[rep.decisao] if chain.resolved else "E"
    limitations = sorted({text_of(a, "en") for v in chain.final.verifications for a in v.assumptions})
    readiness = {"design screening": {"ready": rep.decisao == "APPROVED" and chain.resolved,
                                      "missing": [] if chain.resolved else ["requirements still failing"]}}
    return trust_stages(str(out_dir), checks, {}, {}, tier, readiness, {}, False, limitations)
