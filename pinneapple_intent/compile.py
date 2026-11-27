"""The Engineering Intent Compiler: a sentence in, a typed study contract out — or the questions it needs, or a
refusal that says what is missing.

The compiler is deterministic first. It reads every ``number unit`` pair with the parser ``pp.design`` already uses
(one parser, two consumers), gives each to the slot of the same dimension whose keywords are closest, and fills the
rest from the references the family declares. A value it cannot read and the family cannot default becomes a
question; nothing is ever made up.

A model can be plugged in behind :class:`IntentLLM` — anything with ``extract(text, schema) -> dict``. Its answer is
used only where it can be checked: the family must exist, and every value it proposes must come with a quote that
occurs in the sentence and contains the number. The deterministic reading always runs and wins on conflicts, so the
compiler works with no network and no model at all.
"""
from __future__ import annotations

from typing import Any, Protocol

from pinneapple_design.intent.parse import OutOfScope, fill_values, out_of_scope

from .contract import (
    EXECUTABLE_MODES,
    MODE_SIMULATION,
    MODES,
    Provenance,
    StudyContract,
)
from .family import MIN_SCORE, StudyFamily, get_family, select_family


class IntentLLM(Protocol):
    """What the compiler needs from a model: a structured extraction, and nothing else."""

    def extract(self, text: str, schema: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


# Engineers write "W/m2K" and "Pa s"; the dimensional analyser reads a product only when the factors are separated.
# The compiler rewrites the glued spellings it can rewrite unambiguously before anything is parsed. It does not
# touch "mK", which is a millikelvin: an accidental split there would change the dimension of a sentence silently.
_GLUED = (
    (r"(?i)\bW\s*/\s*\(?\s*m\s*(?:\^?2|²)\s*[·*.]?\s*(?:K|°C|C)\s*\)?", "W/m^2*K"),
    (r"(?i)\bW\s*/\s*m\s*[·*.]?\s*(?:K|°C|C)\b", "W/m*K"),
    (r"(?i)\b(?:J|kJ)\s*/\s*kg\s*[·*.]?\s*(?:K|°C)\b", lambda m: "J/kg*K" if m.group(0)[0] in "Jj" else "kJ/kg*K"),
    (r"(?i)\bPa\s*[·.]\s*s\b", "Pa*s"),
)


def normalise_units(text: str) -> str:
    """Rewrite the glued unit spellings of a request into the products the dimensional analyser reads.

    ``W/m2K`` becomes ``W/(m^2*K)``, ``W/mK`` becomes ``W/(m*K)``, ``Pa·s`` becomes ``Pa*s``; everything else is
    left exactly as written, because rewriting a unit the user did not mean is worse than refusing to read it.
    """
    import re
    out = text
    for pattern, replacement in _GLUED:
        out = re.sub(pattern, replacement, out)
    return out


def schema(family: StudyFamily | None = None) -> dict[str, Any]:
    """The JSON schema-shaped description of what a model may propose, for a prompt or an extraction call."""
    described = [chosen.describe() for chosen in ([family] if family is not None else list(_all_families()))]
    return {
        "type": "object",
        "properties": {
            "family": {"type": "string", "enum": [f["name"] for f in described]},
            "mode": {"type": "string", "enum": list(MODES)},
            "values": {"type": "array", "items": {
                "type": "object",
                "properties": {"slot": {"type": "string"}, "value": {"type": "number"},
                               "unit": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["slot", "value", "unit", "quote"]}},
        },
        "families": described,
    }


def _all_families():
    from .family import families
    return families().values()


def compile_study(text: str, *, answers: dict[str, Any] | None = None, family: str | StudyFamily | None = None,
                  mode: str = MODE_SIMULATION, llm: IntentLLM | None = None, seed: int = 0) -> StudyContract | OutOfScope:
    """Compile a request into a :class:`~pinneapple_intent.contract.StudyContract`.

    * ``answers`` fills the follow-up questions (``{slot: value}``, ``{slot: (value, unit)}`` or ``"70 degC"``).
    * ``family`` forces a family instead of selecting one from the words.
    * ``mode`` is ``simulation`` (analytic, numerical, convergence) or ``physics_ai`` (adds the surrogate rung).
      ``science`` is recognised and refused with what it would need, because the literature stage is not built.
    * ``llm`` is optional; the deterministic compiler runs regardless and wins on conflicts.
    """
    request = (text or "").strip()
    provenance = Provenance.capture(request, seed=seed)
    if not request:
        return OutOfScope(text or "", "no_template", "The request is empty.",
                          "a sentence saying what to simulate, and the numbers it depends on")
    request = normalise_units(request)
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known: {list(MODES)}")
    if mode not in EXECUTABLE_MODES:
        return OutOfScope(request, "needs",
                          "Text-to-science needs a literature stage this release does not have.",
                          "the article or hypothesis to reproduce, and a research stage that searches, cites and "
                          "compares published results; text-to-simulation and text-to-physics-ai run end to end "
                          "today", [], )

    llm_out: dict[str, Any] = {}
    if llm is not None:
        try:
            llm_out = dict(llm.extract(request, schema()) or {})
            provenance = Provenance.capture(request, compiler="llm+deterministic", llm=type(llm).__name__, seed=seed)
        except Exception as exc:  # noqa: BLE001 - a model failure falls back to the deterministic compiler
            llm_out = {}
            provenance = Provenance.capture(request, llm_note=f"{type(exc).__name__}: {str(exc)[:200]}", seed=seed)

    refusal = out_of_scope(request)
    if refusal is not None and refusal[0] == "refused":
        return OutOfScope(request, "refused", refusal[1], "nothing: this request will not be simulated")

    chosen, ranked = _choose_family(request, family, llm_out)
    if chosen is None:
        known = ", ".join(sorted(f.title for f in _all_families()))
        return OutOfScope(request, "no_template", "No study family matches this request.",
                          f"a family for this kind of physics; the installed ones are: {known}. "
                          "docs/guides/intent_to_study.md shows how to add one.", ranked[:3])
    if refusal is not None and ranked and ranked[0][1] < 2 * MIN_SCORE:
        return OutOfScope(request, "needs", f"The request {refusal[1]}", refusal[1].split(": ", 1)[-1], ranked[:3])

    spec = fill_values(chosen, request, ranked=ranked, llm_out=llm_out if llm_out.get("family") in (chosen.name, None)
                       else {})
    spec = _apply_hints(spec, chosen, request)
    contract = StudyContract(
        family=chosen.name, mode=mode, objective=chosen.objective, title=chosen.title, domain=chosen.domain,
        equations=chosen.equations, assumptions=chosen.assumptions, limitations=chosen.limitations,
        references=chosen.references, values=dict(spec.values),
        ladder=chosen.rungs(mode), validation=chosen.validation,
        questions=list(spec.questions), issues=list(spec.issues), unsupported=list(spec.unsupported),
        provenance=provenance,
    )
    if answers:
        contract = contract.answer(chosen.slot_map(), **answers)
    scope = chosen.check_scope(contract)
    if scope:
        return OutOfScope(request, "needs", "; ".join(scope), "; ".join(scope), ranked[:3])
    return contract


def _apply_hints(spec, family: StudyFamily, text: str):
    """Fill the slots the family has an explicit phrase for, overriding the generic reading, and drop the questions
    those values answer."""
    import re

    from pinneapple_design.intent.spec import STATED, SpecError

    slots = family.slot_map()
    for slot_name, pattern in family.hints:
        found = re.search(pattern, text)
        if found is None or slot_name not in slots:
            continue
        slot = slots[slot_name]
        unit = found.groupdict().get("u") or slot.shown_unit()
        try:
            spec.values[slot_name] = slot.quantity(float(found.group("v").replace(",", ".")), unit, STATED,
                                                   source="user", quote=found.group(0))
        except (SpecError, ValueError) as exc:
            spec.issues.append(str(exc))
            continue
        spec.questions = [q for q in spec.questions if q.slot != slot_name]
    spec.questions = [q for q in spec.questions if not (spec.values.get(q.slot) is not None
                                                        and spec.values[q.slot].known)]
    return spec


def _choose_family(request: str, family: str | StudyFamily | None,
                   llm_out: dict[str, Any]) -> tuple[StudyFamily | None, list[tuple[str, float]]]:
    """The family to compile against: forced, matched in the words, or proposed by the model when the words fail."""
    if isinstance(family, StudyFamily):
        return family, [(family.name, 1.0)]
    if isinstance(family, str):
        return get_family(family), [(family, 1.0)]
    chosen, ranked = select_family(request)
    if chosen is None:
        proposed = llm_out.get("family")
        if isinstance(proposed, str):
            try:
                return get_family(proposed), ranked
            except KeyError:
                return None, ranked
    return chosen, ranked


__all__ = ["IntentLLM", "compile_study", "schema"]
