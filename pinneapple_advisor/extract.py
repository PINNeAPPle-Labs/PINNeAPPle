"""Reading a problem description into a brief: rules first, a local language model only for what the words leave
open, and every answer checked against the closed set of fields.

Two readers, in this order:

* :func:`detect` -- explicit keyword and unit rules (English and Portuguese). It is offline, deterministic and
  cheap, and it answers the facts a sentence states outright: "3-D", "transient", "I have the CAD", "200
  simulations", "1 %".
* :func:`read_with_llm` -- a structured extraction through :func:`pinneapple_llm.call_llm` (``provider="ollama"``
  by default: a server on the user's own machine, nothing leaves it). The model may only fill fields that are
  still empty, and only with a value the field allows; anything else lands in ``Brief.rejected`` with the reason.
  It can add a note, never a number the plan uses.

The model never decides anything: the plan's choices come from PINNeAPPle's own deterministic recommendation APIs
(``pinneapple_advisor.compose``). That is the same discipline as ``pinneapple_llm.guardrail`` -- an LLM selects and
parametrises, the physics re-checks.

No model and no server is a working configuration: ``extract(text)`` alone fills what the sentence says and returns
the questions the user still has to answer (:func:`pinneapple_advisor.questions.next_questions`).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from .brief import FIELDS, Brief
from .questions import BY_FIELD, BY_ID

Call = Callable[..., str]

# ------------------------------------------------------------------------------------------------ rules

_NUMBER_WORDS = {
    "um": 1, "uma": 1, "one": 1, "dois": 2, "duas": 2, "two": 2, "três": 3, "tres": 3, "three": 3,
    "cinco": 5, "five": 5, "dez": 10, "ten": 10, "vinte": 20, "twenty": 20, "cem": 100, "hundred": 100,
}

_FAMILY_KEYWORDS = {
    "navier_stokes": ("navier", "stokes", "escoamento", "flow", "cfd", "fluido", "fluid", "aerodin", "aerodynam",
                      "ventilação", "ventilation", "arrasto", "drag", "esteira", "wake"),
    "heat_equation": ("calor", "heat", "térmic", "therm", "temperatura", "temperature", "condução", "conduction"),
    "laplace": ("laplace", "potencial", "potential", "harmônic", "harmonic"),
    "poisson": ("poisson",),
    "wave": ("onda", "wave", "acústic", "acoustic", "sísm", "seism"),
    "helmholtz": ("helmholtz",),
    "linear_elasticity": ("elastic", "estrutur", "structur", "tensão", "stress", "deform", "vibraç", "vibration",
                          "mecânic", "mechanic"),
    "darcy": ("darcy", "poroso", "porous", "permeab", "aquífer", "aquifer"),
    "burgers": ("burgers",),
    "reaction_diffusion": ("reação", "reaction", "difusão", "diffusion", "difus"),
    "sir": ("epidemia", "epidemic", "sir ", "contágio", "contagion"),
    "black_scholes": ("black-scholes", "black scholes", "opç", "option pric"),
    "inviscid_burgers": ("inviscid", "inviscido"),
}

_INVERSE = ("invers", "inverse", "calibrar", "calibrat", "identificar o", "identify the", "estimar o coeficiente",
            "estimate the coefficient", "recover", "recuperar o parâmetro", "parameter identification", "assimil")
_OPTIMIZE = ("otimiz", "optimis", "optimiz", "minimiz", "maximiz", "melhor projeto", "best design", "design space")
_SCREEN = ("triagem", "screen", "muitos casos", "many cases", "monte carlo", "varredura", "sweep")
_HAVE = ("tenho", "temos", "já tenho", "ja tenho", "possuo", "i have", "we have", "there is a", "disponho")


def _has(text: str, *words: str) -> bool:
    return any(w in text for w in words)


def detect(text: str) -> dict[str, Any]:
    """The facts the sentence states outright, as ``field -> value``. Never guesses a field with no evidence."""
    t = " " + text.lower().strip() + " "
    out: dict[str, Any] = {}

    # dimension, written as 3d/3-D/three-dimensional/"em três dimensões"
    if _has(t, "3d", "3-d", "3 d ", "três dimensões", "tres dimensoes", "three-dimensional", "three dimensional",
            "tridimension"):
        out["dimension"] = 3
    elif _has(t, "2d", "2-d", "2 d ", "duas dimensões", "duas dimensoes", "two-dimensional", "two dimensional",
              "bidimension", "placa", "plate", "chapa"):
        out["dimension"] = 2
    elif _has(t, "1d", "1-d", "1 d ", "unidimensional", "one-dimensional", "one dimensional", "barra 1d"):
        out["dimension"] = 1

    if _has(t, "transiente", "transient", "unsteady", "dependente do tempo", "time-dependent", "time dependent",
            "no tempo", "over time", "dinâmic", "dynamic"):
        out["time_dependent"] = True

    if _has(t, "turbulent", "turbulento", "número de reynolds alto", "alto reynolds", "high reynolds",
            "esteira turbulenta"):
        out["turbulent"] = True

    if _has(t, "não linear", "nao linear", "nonlinear", "não-linear", "non-linear", "plastic", "plástic", "contato",
            "contact", "radiação", "radiation", "fase", "phase change", "mudança de fase"):
        out["nonlinear"] = True

    # Possession is stated in the first person; an instruction ("gere uma malha") is not possession.
    if _has(t, ".stl", ".step", ".iges", ".cad", "cadquery", "freecad", "solidworks", "stl", "step", "iges"):
        out.setdefault("geometry_formats", [])
        for fmt, keys in (("stl", (".stl", "stl")), ("step", (".step", ".iges", "step", "iges")),
                          ("cad", ("cadquery", "freecad", "solidworks", ".cad"))):
            if _has(t, *keys) and fmt not in out["geometry_formats"]:
                out["geometry_formats"].append(fmt)
    if (_has(t, "geometria", "geometry", "cad", "peça", "part", "step", "stl", "iges") and _has(t, *_HAVE)
            and not _has(t, "não tenho geometria", "sem geometria", "no geometry", "não tenho o cad")):
        out["geometry_present"] = True
    if _has(t, "malha pronta", "malha existente", "já tenho a malha", "tenho a malha", "i have a mesh",
            "mesh file", ".msh", "existing mesh", "openfoam case", "caso openfoam"):
        out["mesh_present"] = True
    if _has(t, "dataset", "dados de simulação", "simulation data", "resultados de simulação", "snapshots",
            "já rodamos", "already ran", "banco de dados de simulações") and _has(t, *_HAVE, "dataset"):
        out["dataset_present"] = True
    if _has(t, "não tenho dataset", "sem dataset", "no dataset", "from scratch", "do zero"):
        out["dataset_present"] = False

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(k|mil|thousand|m|mi|million|milh(?:ão|oes|ões)?)?\s*"
                  r"(simulaç|simulation|casos|case|snapshot|runs?|amostras|samples)", t)
    if m:
        value = float(m.group(1).replace(",", "."))
        unit = (m.group(2) or "").strip()
        if unit in ("k", "mil", "thousand"):
            value *= 1e3
        elif unit in ("m", "mi", "million", "milh", "milhão", "milhões", "milhoes"):
            value *= 1e6
        out.setdefault("n_simulations_present", int(value))
        if "dataset_present" not in out:
            out["dataset_present"] = True

    if _has(t, "família de geometrias", "family of geometries", "várias geometrias", "different geometries",
            "geometrias diferentes", "otimização de forma", "shape optimis", "shape optimiz", "topology",
            "geometria varia", "geometry varies", "geometries vary"):
        out["geometry_varies"] = True
    if _has(t, "paramétric", "parametric", "varredura", "sweep", "doe", "design of experiments", "vários parâmetros",
            "different parameters", "parâmetros variam", "parameters vary", "variar", "variação", "variações",
            "varying", "vary ", "several values", "diferentes valores", "surrogate", "substituto", "treinar um"):
        out["parameter_varies"] = True
    if _has(t, "surrogate", "substituto", "treinar um modelo", "train a model", "neural operator", "operador"):
        out.setdefault("output_kind", "grid_field")
    if _has(t, *_INVERSE):
        out["goal"] = "inverse"
    elif _has(t, *_OPTIMIZE):
        out["goal"] = "optimization"
    elif _has(t, *_SCREEN):
        out["goal"] = "screening"

    # a target written as a percentage ("erro de 2 %") is the accuracy the study must reach
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", t)
    if m and _has(t, "erro", "error", "precis", "accura", "tolerân", "toleran"):
        out["accuracy_target"] = max(float(m.group(1).replace(",", ".")) / 100.0, 1e-6)

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(k|mil|thousand|m|mi|million|milh(?:ão|oes|ões)?)?\s*"
                  r"(células|c[eé]lulas|cells|elementos|elements|nós|nodes)", t)
    if m:
        value = float(m.group(1).replace(",", "."))
        unit = (m.group(2) or "").strip()
        if unit in ("k", "mil", "thousand"):
            value *= 1e3
        elif unit in ("m", "mi", "million", "milh", "milhão", "milhões", "milhoes"):
            value *= 1e6
        out["approx_cells"] = int(value)

    if _has(t, "gpu", "cuda", "nvidia", "metal", "mps"):
        out["gpu"] = True
    m = re.search(r"(\d+)\s*(núcleos|nucleos|cores|cpus?)", t)
    if m:
        out["cores"] = int(m.group(1))

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(s|seg|segundos|seconds|min|minutos|minutes|h|horas|hours)\b", t)
    if m and _has(t, "por simulação", "per simulation", "cada simulação", "each simulation", "leva", "takes"):
        value = float(m.group(1).replace(",", "."))
        unit = m.group(2)
        factor = {"s": 1, "seg": 1, "segundos": 1, "seconds": 1, "min": 60, "minutos": 60, "minutes": 60,
                  "h": 3600, "horas": 3600, "hours": 3600}.get(unit, 1)
        out["solve_seconds_measured"] = value * factor

    family = _family(t)
    if family:
        out["physics_family"] = family
    return out


def _family(text: str) -> str | None:
    """The PDE family: PINNeAPPle's own keyword scorer first (``identify_pde``), the explicit words above as a
    tie-break. ``custom`` is never guessed -- it is a deliberate answer to the question."""
    scores: dict[str, float] = {}
    try:                                        # pinneapple_physics pulls torch; the advisor works without it
        from pinneapple_physics.pde_environment.capabilities import identify_pde
        for kind, score in identify_pde(text):
            scores[kind] = max(scores.get(kind, 0.0), float(score))
    except Exception:                           # noqa: BLE001 - an import failure only costs the extra scorer
        pass
    for kind, words in _FAMILY_KEYWORDS.items():
        hits = sum(1 for w in words if w in text)
        if hits:
            scores[kind] = scores.get(kind, 0.0) + 2.0 * hits
    if not scores:
        return None
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] >= 2.0 else None


# ------------------------------------------------------------------------------------------------ the model

_SYSTEM = ("You extract study facts for PINNeAPPle. You answer with one JSON object and nothing else. You never "
           "invent a fact the text does not state: a field you cannot fill is left out.")


def brief_prompt(text: str, unknown: list[str]) -> str:
    """The extraction prompt: only the fields still open, each with the values it accepts."""
    lines = []
    for name in unknown:
        spec = FIELDS[name]
        if spec["kind"] == "choice":
            allowed = ", ".join(spec["choices"])
        elif spec["kind"] == "list":
            allowed = ", ".join(spec["choices"])
        elif spec["kind"] == "bool":
            allowed = "true or false"
        elif spec["kind"] == "int":
            allowed = f"whole number between {spec.get('min')} and {spec.get('max')}"
        else:
            allowed = f"number between {spec.get('min')} and {spec.get('max')}"
        lines.append(f'- "{name}" ({spec["kind"]}): {allowed}')
    return (f"Problem description:\n<<<\n{text.strip()[:4000]}\n>>>\n\n"
            f"Fields to fill, with the only values they accept:\n" + "\n".join(lines) +
            "\n\nAnswer with JSON: {\"fields\": {<field>: <value>, ...}, \"notes\": [\"short facts worth keeping, "
            "in the language of the description\"]}. Omit a field the text does not state.")


def _extract_json(raw: str) -> dict:
    raw = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S)
    if m:
        raw = m.group(1)
    if not raw.startswith("{"):
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            raw = raw[i:j + 1]
    return json.loads(raw)


def read_with_llm(brief: Brief, text: str, *, call: Call | None = None, provider: str = "ollama",
                  model: str = "", **llm_kwargs: Any) -> Brief:
    """Fill the brief's still-empty fields with a language model, validated field by field.

    ``call`` is ``pinneapple_llm.call_llm`` by default; a test (or the API layer) passes its own callable taking
    the prompt as first argument and returning the raw text. A failure of the model is recorded as a note, never
    raised: the offline reading is already a usable brief.
    """
    unknown = [f for f in brief.unknown() if f in BY_FIELD]
    if not unknown:
        return brief
    if call is None:
        from pinneapple_llm._dispatch import call_llm as call
    label = f"{provider}/{model or 'default'}"
    try:
        raw = call(brief_prompt(text, unknown), provider=provider, model=model or None, system=_SYSTEM,
                   json_mode=True, module="pinneapple_advisor", **llm_kwargs)
        payload = _extract_json(raw)
    except Exception as exc:                    # noqa: BLE001 - a model that is down must not break the advice
        brief.notes.append(f"language model {label} did not answer ({type(exc).__name__}: {str(exc)[:200]}); "
                           f"the offline reading and the questions below are still complete")
        return brief
    fields = dict(payload.get("fields") or {})
    for name, value in fields.items():
        if name not in FIELDS:
            brief.rejected.append({"field": name, "value": value, "source": f"llm:{label}",
                                   "why": "unknown field"})
            continue
        if name not in unknown:
            continue                            # a rule already answered this; the rule wins
        brief.try_set(name, value, f"llm:{label}")
    for note in list(payload.get("notes") or [])[:5]:
        brief.notes.append(str(note)[:300])
    return brief


# ------------------------------------------------------------------------------------------------ answers

_TRUE = ("sim", "yes", "y", "true", "tenho", "i have", "we have", "com certeza", "positivo", "1")
_FALSE = ("não", "nao", "no", "n", "false", "nenhum", "nenhuma", "sem", "0", "negativo")


def answer(question_id: str, raw: Any, *, call: Call | None = None, provider: str = "ollama",
           model: str = "", **llm_kwargs: Any) -> Any:
    """Turn one answer into the value its question accepts.

    A choice answer is matched against the question's own choices (value first, then the labels in either
    language); a yes/no answer against the usual words in both languages; a number is read directly. When nothing
    matches and a model is available, the model maps the sentence onto the question's closed set -- and its answer
    is checked the same way, so a bad mapping is refused rather than passed on.
    """
    q = BY_ID.get(question_id) or BY_FIELD.get(question_id)
    if q is None:
        raise KeyError(f"no question {question_id!r}; known: {', '.join(BY_ID)}")
    text = str(raw).strip()
    low = text.lower()

    if q.kind == "bool":
        if isinstance(raw, bool):
            return raw
        if any(w == low or low.startswith(w + " ") or w in low.split() for w in _TRUE):
            return True
        if any(w == low or low.startswith(w + " ") or w in low.split() for w in _FALSE):
            return False
    elif q.kind in ("int", "number"):
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return raw
        m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(k|mil|thousand|m|mi|million|milh(?:ão|ões|oes)?)?", low)
        if m:
            value = float(m.group(1).replace(",", "."))
            unit = (m.group(2) or "").strip()
            if unit in ("k", "mil", "thousand"):
                value *= 1e3
            elif unit in ("m", "mi", "million", "milh", "milhão", "milhões", "milhoes"):
                value *= 1e6
            if "%" in low:
                value /= 100.0
            return int(value) if q.kind == "int" else value
    elif q.kind == "choice":
        for choice in q.choices:
            if low == choice["value"] or low == choice["pt"].lower() or low == choice["en"].lower():
                return choice["value"]
        for choice in q.choices:
            label = choice["pt"].lower()
            if label in low or choice["en"].lower() in low:
                return choice["value"]
        # "3 dimensões", "three", "opção 2": a lone number that matches one of the choice values
        number = re.search(r"\b(\d+)\b", low)
        if number and any(c["value"] == number.group(1) for c in q.choices):
            return number.group(1)
    elif q.kind == "list":
        picked = [c["value"] for c in q.choices if c["value"] in low]
        if picked:
            return picked

    if not call:
        return raw                              # the caller validates, and its error names what was expected
    prompt = (f"Question: {q.en}\nAccepted answers: {json.dumps([c['value'] for c in q.choices] or q.kind)}\n"
              f"Answer given: {text!r}\nReturn JSON {{\"value\": <one accepted value>}}.")
    out = _extract_json(call(prompt, provider=provider, model=model or None, system=_SYSTEM, json_mode=True,
                             module="pinneapple_advisor", **llm_kwargs))
    return out.get("value", raw)


def extract(text: str, *, call: Call | None = None, provider: str = "ollama", model: str = "",
            brief: Brief | None = None, **llm_kwargs: Any) -> Brief:
    """The brief for a description: the rules, then (when ``call`` or a model is asked for) the model for the rest.

    ``call=None`` means "use :func:`pinneapple_llm.call_llm` with ``provider``"; pass ``call=False`` to stay
    strictly offline (the tests do this, and so does a machine with no Ollama).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("describe the problem in a sentence or more")
    brief = brief or Brief()
    brief.description = brief.description or text.strip()
    brief.merge(detect(text), source="rule")
    if call is False:
        return brief
    return read_with_llm(brief, text, call=call, provider=provider, model=model, **llm_kwargs)


__all__ = ["detect", "extract", "read_with_llm", "answer", "brief_prompt"]
