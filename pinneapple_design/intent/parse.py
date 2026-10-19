"""Intent -> RequirementSpec: pick a template, read the numbers with their units, fill the template's slots.

The deterministic parser needs no network: it finds every ``number unit`` pair (``2 kN``, ``40W``, ``85 °C``,
``30 cm``), checks the unit with :mod:`pinneapple_data.physical_units`, and gives each to the slot of the same
dimension whose keywords are closest in the sentence. Phrases map to values through the slots' cues ("still air" ->
0 m/s). Nothing is guessed: a value it cannot place is reported as unused, a critical slot it cannot fill becomes a
question.

An LLM can be plugged in behind :class:`IntentLLM` (anything with ``extract(text, schema) -> dict``). Its answer is
used only where it can be checked: the template must exist, every value must come with a quote that occurs in the
text and contains the number, and the unit must have the slot's dimension. The deterministic reading wins on
conflicts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from pinneapple_data.physical_units import try_parse_unit

from . import materials as M
from .spec import ANSWERED, MISSING, STATED, Question, RequirementSpec, SpecError
from .template import ProblemTemplate

# ============================================================================================ registry
_REGISTRY: dict[str, ProblemTemplate] = {}


def register(template: ProblemTemplate) -> ProblemTemplate:
    """Add a template (an instance) to the registry used by :func:`formulate`."""
    if not template.name:
        raise ValueError("a template needs a name")
    _REGISTRY[template.name] = template
    return template


def templates() -> dict[str, ProblemTemplate]:
    _load_builtin()
    return dict(_REGISTRY)


def get_template(name: str) -> ProblemTemplate:
    _load_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"unknown template {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def _load_builtin() -> None:
    if not _REGISTRY:
        from .templates import builtin
        for t in builtin():
            register(t)


# Requests the library recognises but has no template for, and what would be needed. Order matters: the first
# matching entry gives the answer.
OUT_OF_SCOPE: list[tuple[tuple[str, ...], str, str]] = [
    (("weapon", "gun", "rifle", "pistol", "explosive", "bomb", "missile", "grenade", "warhead"), "refused",
     "Designing weapons or explosives is not something this tool does."),
    (("fusion", "fission", "nuclear", "reactor core", "radioactive"), "needs",
     "needs nuclear-physics models (neutronics, plasma) and licensing data; no template covers them."),
    (("drug", "medicine", "vaccine", "implant", "prosthesis", "prosthetic", "surgical", "medical device"), "needs",
     "needs biomedical data and regulatory evidence (biocompatibility, clinical); no template covers them."),
    (("motor", "generator", "transformer", "inductor", "antenna", "electromagnet", "magnet", "magnetic", "coil", "solenoid",
      "pcb trace"), "needs",
     "needs an electromagnetics design template: the interior-PM machine FEA and drive models exist "
     "(pinneapple_simulation.electromagnetics.machines), but no template sizes a machine from a request yet, and "
     "it would also need winding current-density, thermal and demagnetisation limits, which those models do not "
     "check."),
    (("battery", "catalyst", "chemical reactor", "combustion", "fuel cell", "electrolyser", "molecule"), "needs",
     "needs chemistry and reaction-kinetics models; no template covers them."),
    (("bridge", "building", "skyscraper", "dam", "tower crane", "stadium", "house"), "needs",
     "needs code-based civil design (Eurocode / AISC load combinations, site data) and a licensed engineer; the "
     "structural template covers small machine brackets only."),
    (("rocket", "spacecraft", "satellite", "orbit", "jet engine", "turbine engine", "supersonic", "hypersonic"),
     "needs", "needs compressible-flow, propulsion and trajectory models; the fluid template covers low-speed "
     "glider wings only."),
    (("app", "website", "software", "algorithm", "poem", "story", "logo"), "needs",
     "is not a physical design: this tool designs parts from physics (structures, heat, flow, dynamics)."),
]


@dataclass
class OutOfScope:
    """The request cannot be designed here: ``reason`` says why and ``needs`` what would be required."""
    text: str
    kind: str                          # "refused" | "needs" | "no_template"
    reason: str
    needs: str
    closest: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"status": "out_of_scope", "kind": self.kind, "text": self.text, "reason": self.reason,
                "needs": self.needs, "closest_templates": [list(c) for c in self.closest]}

    def __str__(self) -> str:
        return f"Out of scope ({self.kind}): {self.reason}"


# ============================================================================================ text helpers
_NUM = r"[-+]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_TOKEN = re.compile(r"[A-Za-z°µμ%Ω²³/·*^\-]+[0-9]?|\S")
_WORD = re.compile(r"[a-z0-9°]+(?:-[a-z]+)?")


@dataclass
class Mention:
    value: float
    unit: str                         # "" when no unit follows
    start: int                        # character span of "number unit"
    end: int
    quote: str


def _norm(text: str) -> str:
    return (text.replace(" ", " ").replace("º", "°").replace("˚", "°").replace("’", "'")
            .replace("–", "-").replace("×", "x"))


def find_quantities(text: str) -> list[Mention]:
    """Every number in ``text`` with the unit that follows it (the longest run of up to three tokens that parses as a
    unit, e.g. ``m / s``, ``deg C``, ``W/m2K``). Numbers glued to letters (``6061``, ``NACA2412``) are skipped."""
    t = _norm(text)
    out = []
    for m in re.finditer(_NUM, t):
        a, b = m.span()
        if a > 0 and (t[a - 1].isalpha() or t[a - 1] in "-_/" and a > 1 and t[a - 2].isalpha()):
            continue
        if b < len(t) and t[b].isdigit():
            continue
        try:
            v = float(m.group().replace(",", "."))
        except ValueError:
            continue
        rest = t[b:]
        unit, uend = "", b
        cands = []
        toks = list(re.finditer(r"\s*([^\s,;:()]+)", rest[:40]))
        for k in range(min(3, len(toks)), 0, -1):
            txt = "".join(x.group(0) for x in toks[:k]).strip().rstrip(".")
            cands.append((txt, b + toks[k - 1].end()))
        for txt, end in cands:
            u = _unit(txt)
            if u is not None:
                unit, uend = u, end
                break
        # a number directly followed by a word that is not a unit has no unit
        out.append(Mention(v, unit, a, uend, t[a:uend].strip()))
    return out


_UNIT_ALIASES = {"c": "°C", "degc": "°C", "deg c": "°C", "degrees c": "°C", "° c": "°C", "celsius": "°C",
                 "degrees celsius": "°C", "f": "°F", "kgs": "kg", "kilos": "kg", "kilo": "kg", "grams": "g",
                 "watts": "W", "watt": "W", "newtons": "N", "kilonewtons": "kN", "meters": "m", "metres": "m",
                 "meter": "m", "metre": "m", "centimetres": "cm", "centimeters": "cm", "millimetres": "mm",
                 "millimeters": "mm", "inches": "in", "inch": "in", "kph": "km/h", "mph": "mi/h", "percent": "%"}
_NOT_UNITS = {"a", "an", "and", "at", "in", "s", "for", "to", "or", "on", "x", "times", "of", "by", "per", "is",
              "it", "t", "h", "min", "l", "v", "j", "pa"}


def _unit(txt: str) -> str | None:
    s = txt.strip().strip(".").strip()
    low = s.lower()
    if low in _UNIT_ALIASES:
        return _UNIT_ALIASES[low]
    if low in _NOT_UNITS and s not in ("Pa", "V", "J", "h", "min", "s", "in"):
        return None
    if s in ("h", "min", "s") or s == "in":
        return None                   # too ambiguous in free text ("2 s" is rare, "in" is a preposition)
    if s == "%":
        return "%"
    if not re.fullmatch(r"[A-Za-z°µμ%Ω²³/·*^\-0-9() ]+", s) or s[0].isdigit():
        return None
    u = try_parse_unit(s)
    if u is None:
        return None
    if len(s) > 1 and s.isalpha() and s.islower() and s not in ("kg", "mm", "cm", "km", "kw", "mw", "kn", "mg",
                                                                  "nm", "um", "kpa", "mpa", "gpa", "rpm", "lbf",
                                                                  "lb", "ft", "mi", "g", "w", "n", "m", "k"):
        return None                   # English words that happen to be unit names ("day", "bar"...)
    return s


def _words_with_pos(text: str) -> list[tuple[str, int]]:
    return [(m.group(), m.start()) for m in _WORD.finditer(text.lower())]


def _kw_hit(text_low: str, kw: str) -> list[int]:
    return [m.start() for m in re.finditer(rf"(?<![a-z]){re.escape(kw.lower())}", text_low)]


# ============================================================================================ LLM interface
class IntentLLM(Protocol):
    def extract(self, text: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class CallLLM:
    """Adapter from :func:`pinneapple_llm.call_llm` (Anthropic / OpenAI / Ollama) to :class:`IntentLLM`."""

    def __init__(self, provider: str = "anthropic", model: str | None = None, **kw: Any):
        self.provider, self.model, self.kw = provider, model, kw

    def extract(self, text: str, schema: dict[str, Any]) -> dict[str, Any]:
        from pinneapple_llm import call_llm

        system = ("You turn an engineering design request into JSON. Use only numbers written in the request. "
                  "For every value give the exact quote it comes from. Answer with JSON only, matching: "
                  + json.dumps(schema))
        out = call_llm(text, provider=self.provider, model=self.model, system=system, json_mode=True, **self.kw)
        return json.loads(out) if isinstance(out, str) else dict(out)


def llm_schema(tpls: dict[str, ProblemTemplate]) -> dict[str, Any]:
    return {"template": {"one of": sorted(tpls) + ["none"]},
            "values": [{"slot": "slot name of the chosen template", "value": "number or choice",
                        "unit": "unit as written", "quote": "exact words from the request"}],
            "templates": {n: {"title": t.title, "slots": {s.name: f"{s.label} [{s.shown_unit()}]" for s in t.slots}}
                          for n, t in tpls.items()}}


# ============================================================================================ parsing
def select_template(text: str, tpls: dict[str, ProblemTemplate] | None = None) -> list[tuple[str, float]]:
    """Templates ranked by keyword score (sum of the weights of the keywords found in the text)."""
    tpls = tpls if tpls is not None else templates()
    low = _norm(text).lower()
    scores = []
    for name, t in tpls.items():
        s = sum(w for kw, w in t.keywords.items() if _kw_hit(low, kw))
        scores.append((name, float(s)))
    return sorted(scores, key=lambda x: -x[1])


def out_of_scope(text: str) -> tuple[str, str] | None:
    low = _norm(text).lower()
    for words, kind, why in OUT_OF_SCOPE:
        for w in words:
            if re.search(rf"(?<![a-z]){re.escape(w)}(?:s|es)?(?![a-z])", low):
                return kind, f"'{w}': {why}" if kind != "refused" else why
    return None


MIN_SCORE = 1.0


def formulate(text: str, *, llm: IntentLLM | None = None, answers: dict[str, Any] | None = None,
              template: str | None = None) -> RequirementSpec | OutOfScope:
    """Turn a request into a :class:`RequirementSpec` (or an :class:`OutOfScope` with what is needed).

    ``answers``: values for follow-up questions, ``{slot: value}`` (slot unit) or ``{slot: (value, unit)}``.
    ``template``: force a template instead of selecting one from the text.
    """
    tpls = templates()
    text_n = _norm(text).strip()
    if not text_n:
        return OutOfScope(text, "no_template", "The request is empty.", "a sentence saying what to design")
    ranked = select_template(text_n, tpls)
    llm_out: dict[str, Any] = {}
    parser = "template"
    if llm is not None:
        try:
            llm_out = dict(llm.extract(text_n, llm_schema(tpls)) or {})
            parser = "llm+template"
        except Exception as e:  # noqa: BLE001 - an LLM failure falls back to the deterministic parser
            llm_out = {"error": f"{type(e).__name__}: {e}"}
    oos = out_of_scope(text_n)
    if template is None:
        best, score = ranked[0] if ranked else (None, 0.0)
        llm_t = llm_out.get("template")
        if (score < MIN_SCORE or oos) and llm_t in tpls and score < 2 * MIN_SCORE and not oos:
            best, score = llm_t, MIN_SCORE
        if oos and oos[0] == "refused":
            return OutOfScope(text, "refused", oos[1], "nothing: this request will not be designed",
                              ranked[:3])
        if oos and score < 2 * MIN_SCORE:
            return OutOfScope(text, "needs", f"The request {oos[1]}", oos[1].split(": ", 1)[-1], ranked[:3])
        if score < MIN_SCORE:
            names = ", ".join(f"{t.title}" for t in tpls.values())
            return OutOfScope(text, "no_template",
                              "No problem template matches this request.",
                              f"a template for this kind of design; the installed ones are: {names}. See the "
                              "template-authoring guide to add one.", ranked[:3])
        template = best
    t = tpls[template]
    spec = _fill(t, text_n, ranked, llm_out if llm_out.get("template") in (template, None) else {})
    spec.parser = parser
    if "error" in llm_out:
        spec.issues.append(f"LLM extraction failed ({llm_out['error']}); the template parser was used alone")
    scope = t.check_scope(spec)
    if scope:
        return OutOfScope(text, "needs", "; ".join(scope), "; ".join(scope), ranked[:3])
    if answers:
        spec = spec.answer(t.slot_map(), **answers)
        _refresh_questions(t, spec)
    return spec


def _fill(t: ProblemTemplate, text: str, ranked, llm_out: dict[str, Any]) -> RequirementSpec:
    low = text.lower()
    slots = t.slot_map()
    spec = RequirementSpec(text=text, template=t.name, match_score=dict(ranked).get(t.name, 0.0))
    mentions = find_quantities(text)
    # ---- cues (phrases with a value)
    for s in t.slots:
        for phrase, val in s.cues:
            if _kw_hit(low, phrase) and s.name not in spec.values:
                unit = "" if s.is_choice else s.unit
                try:
                    spec.values[s.name] = s.quantity(val, unit, STATED, source="user", quote=phrase)
                except SpecError as e:
                    spec.issues.append(str(e))
    # ---- materials named in the text
    named = M.find(text)
    for s in t.slots:
        if s.is_choice and s.role == "material" and named and s.name not in spec.values:
            ok = [k for k in named if k in s.choices]
            if ok:
                spec.values[s.name] = s.quantity(ok[0], "", STATED, source="user", quote=ok[0])
            else:
                spec.issues.append(f"{s.label}: {', '.join(M.get(k).label for k in named)} has no cited data for "
                                   f"this template (choices: {', '.join(M.get(k).label for k in s.choices)})")
    # ---- numbers -> slots: score every (mention, slot) pair by keyword proximity
    pairs = []
    for i, mt in enumerate(mentions):
        for s in t.slots:
            if s.name in spec.values or s.is_choice:
                continue
            if mt.unit and not s.accepts(mt.unit):
                continue
            if not mt.unit and s.unit not in ("",):
                continue
            score = 0.0
            for kw in s.keywords:
                for p in _kw_hit(low, kw):
                    if p < mt.start:
                        d = len(_WORD.findall(low[p + len(kw):mt.start]))
                        score = max(score, 1.0 / (1 + d) if d <= 6 else 0.0)
                    else:
                        d = len(_WORD.findall(low[mt.end:p]))
                        score = max(score, 0.8 / (1 + d) if d <= 4 else 0.0)
            if not mt.unit and score < 0.3:
                continue              # a bare number needs a keyword right next to it
            if score == 0.0:
                same = [x for x in t.slots if not x.is_choice and x.name not in spec.values and mt.unit
                        and x.accepts(mt.unit)]
                score = 0.05 if len(same) == 1 else 0.0
            if score > 0:
                pairs.append((score, i, s.name))
    used_m, used_s = set(), set()
    for _score, i, name in sorted(pairs, key=lambda x: -x[0]):
        if i in used_m or name in used_s:
            continue
        mt, s = mentions[i], slots[name]
        try:
            spec.values[name] = s.quantity(mt.value, mt.unit, STATED, source="user", quote=mt.quote)
            used_m.add(i)
            used_s.add(name)
        except SpecError as e:
            spec.issues.append(str(e))
            used_m.add(i)
            used_s.add(name)
    for i, mt in enumerate(mentions):
        if i not in used_m and mt.unit:
            spec.unsupported.append(f"'{mt.quote}' was not used: no value of that kind in the {t.title} template")
    # ---- LLM values, checked against the text
    for item in llm_out.get("values", []) or []:
        try:
            name, val, unit, quote = item["slot"], item["value"], str(item.get("unit", "")), str(item["quote"])
        except (KeyError, TypeError):
            continue
        if name not in slots or name in spec.values:
            continue
        if not quote or _norm(quote).lower() not in low:
            spec.issues.append(f"LLM value for {slots[name].label} rejected: its quote is not in the request")
            continue
        if not slots[name].is_choice and str(val).split(".")[0].lstrip("-") not in quote:
            spec.issues.append(f"LLM value for {slots[name].label} rejected: the number is not in its quote")
            continue
        try:
            spec.values[name] = slots[name].quantity(val, unit, STATED, source="user (read by LLM)", quote=quote)
        except (SpecError, ValueError) as e:
            spec.issues.append(f"LLM value rejected: {e}")
    # ---- objectives
    objs = [o for o in t.objectives if any(_kw_hit(low, k) for k in o.keywords)]
    spec.objectives = objs or t.default_objectives()
    for word, what in (("cheap", "cost"), ("cost", "cost"), ("price", "cost"), ("quiet", "noise"),
                       ("beautiful", "appearance"), ("pretty", "appearance"), ("recyclable", "recyclability")):
        if _kw_hit(low, word):
            spec.unsupported.append(f"'{word}': no {what} model in this template, so it is not optimised")
    # ---- defaults and questions
    for s in t.slots:
        if s.name not in spec.values:
            spec.values[s.name] = s.default_quantity()
    _refresh_questions(t, spec)
    return spec


def _refresh_questions(t: ProblemTemplate, spec: RequirementSpec) -> None:
    asked = {q.slot for q in spec.questions}
    keep = []
    for q in spec.questions:
        if spec.values.get(q.slot) is not None and spec.values[q.slot].provenance in (STATED, ANSWERED):
            continue
        keep.append(q)
    for s in t.slots:
        v = spec.values.get(s.name)
        if s.critical and (v is None or v.provenance == MISSING) and s.name not in asked:
            keep.append(Question(s.name, s.question or f"What is the {s.label}?", s.shown_unit(),
                                 tuple(s.choices), why="needed by the physics and has no safe default"))
    keep = [q for q in keep if q.slot not in {s.name for s in t.slots if not s.critical}]
    have = {q.slot for q in keep}
    keep += [q for q in t.extra_questions(spec) if q.slot not in have]
    spec.questions = keep
