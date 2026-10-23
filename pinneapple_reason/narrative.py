"""The chain told in words: one paragraph per version, each claim followed by the number that proves it."""
from __future__ import annotations

from .chain import RESOLVED, Chain, _fmt, text_of

_L = {
    "pt": {"v": "Versão", "fails": "falha", "proof": "prova", "because": "Por isso", "chosen": "escolhida",
           "tried": "Opções testadas", "infeasible": "inviável", "fixes": "resolve", "not_fix": "não resolve",
           "cost": "custo", "ok": "Todos os requisitos passam.", "unresolved": "Sem solução na biblioteca:",
           "max": "Parou no número máximo de passos.", "req": "Requisitos", "final": "Versão final"},
    "en": {"v": "Version", "fails": "fails", "proof": "proof", "because": "So", "chosen": "chosen",
           "tried": "Options tried", "infeasible": "infeasible", "fixes": "fixes", "not_fix": "does not fix",
           "cost": "cost", "ok": "Every requirement passes.", "unresolved": "No solution in the library:",
           "max": "Stopped at the maximum number of steps.", "req": "Requirements", "final": "Final version"},
}


def option_line(o, lang: str) -> str:
    L = _L[lang]
    if not o.feasible:
        return f"{o.countermeasure}: {L['infeasible']} ({text_of(o.reason, lang)})"
    c = o.cost
    return (f"{o.countermeasure}: {L['fixes'] if o.fixes_target else L['not_fix']} "
            f"(margin {_fmt(o.target_margin_before)} -> {_fmt(o.target_margin_after)}), "
            f"{L['cost']} {_fmt(c.get('quantity', 0))} {c.get('unit', '')} x {_fmt(c.get('unit_cost', 0))} = "
            f"{_fmt(c.get('index', 0))}" + (f"; breaks {', '.join(o.regressions)}" if o.regressions else ""))


def explain(chain: Chain, lang: str = "pt") -> str:
    L = _L["pt" if lang == "pt" else "en"]
    lines = [text_of(chain.title, lang), "", L["req"] + ":"]
    lines += [f"  {r.id}: {text_of(r.text, lang)} [{', '.join(r.modes)}]" for r in chain.requirements]
    for s in chain.steps:
        els = ", ".join(s.concept.elements) or "-"
        lines += ["", f"{L['v']} {s.version} ({els}):"]
        if s.finding is None:
            lines.append(f"  {L['ok']}")
            for v in s.verifications:
                lines.append(f"  - {v.requirement}/{v.mode}: {_fmt(v.demand)} <= {_fmt(v.capacity)} {v.unit}")
            continue
        v = s.finding.verification
        lines.append(f"  {text_of(s.finding.title, lang)} {L['fails']}: {text_of(v.mechanism, lang)}")
        lines.append(f"  [{L['proof']}: {text_of(v.demand_label, lang)} {_fmt(v.demand)} {v.unit} > "
                     f"{text_of(v.capacity_label, lang)} {_fmt(v.capacity)} {v.unit}; {v.method}]")
        if s.options:
            lines.append(f"  {L['tried']}:")
            lines += [f"    - {option_line(o, lang)}" for o in s.options]
        cm = s.countermeasure
        if cm is not None:
            lines.append(f"  {L['because']}: {text_of(cm.title, lang)} — {text_of(cm.effect, lang)}")
            if cm.changes:
                lines.append("  -> " + ", ".join(f"{k} = {_fmt(x) if isinstance(x, float) else x}"
                                                for k, x in cm.changes.items()))
            lines.append(f"  ({L['chosen']}: {text_of(s.why, lang)})")
        elif s.why:
            lines.append(f"  {L['unresolved']} {text_of(s.why, lang)}")
    if chain.status != RESOLVED and chain.final.finding is not None and chain.final.countermeasure is None \
            and not chain.final.why:
        lines += ["", L["max"]]
    return "\n".join(lines)
