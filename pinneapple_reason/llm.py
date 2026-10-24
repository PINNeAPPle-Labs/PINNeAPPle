"""An LLM as an advisor of the reasoning: it may propose candidate countermeasures and a first concept, never a number
that decides anything.

``LLMAdvisor`` wraps any object with ``extract(text, schema) -> dict`` (the
:class:`pinneapple_design.intent.IntentLLM` protocol; :class:`pinneapple_design.intent.CallLLM` reaches Anthropic,
OpenAI or a local Ollama model through :func:`pinneapple_llm.call_llm`). What it returns is filtered:

* countermeasures: only names that exist in the library are kept; the engine applies them and computes their effect,
  cost and side effects like any other option (the option records ``proposed_by="llm"``);
* concept parameters: only parameters some failure mode of the requirements declares (``Need``) are kept, each
  marked ``"llm proposal (premise)"`` in ``Concept.sources`` so the report shows where it came from.

Without an LLM nothing changes: the library's own countermeasures are always tried.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

from .chain import Concept, Finding, Requirement, text_of
from .library import LIBRARY, Library

__all__ = ["LLMAdvisor", "ExtractLLM"]


class ExtractLLM(Protocol):
    def extract(self, text: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class LLMAdvisor:
    def __init__(self, llm: ExtractLLM, max_proposals: int = 3):
        self.llm = llm
        self.max_proposals = int(max_proposals)
        self.log: list[dict[str, Any]] = []

    def propose_countermeasures(self, finding: Finding, concept: Concept, library: Library = LIBRARY) -> list[str]:
        catalog = {c.name: {"title": text_of(c.title, "en"), "effect": text_of(c.effect, "en"),
                            "addresses": list(c.addresses)} for c in library.countermeasures.values()}
        text = (f"Failure found: {text_of(finding.title, 'en')}. Mechanism: "
                f"{text_of(finding.verification.mechanism, 'en')} Concept elements: {concept.elements}. "
                "Which countermeasures from the catalog could act on this mechanism (also ones listed for other "
                "modes, if the physics allows)? Names only.")
        schema = {"countermeasures": ["name from the catalog"], "catalog": catalog}
        out = self.llm.extract(text, schema)
        names = [n for n in (out.get("countermeasures") or []) if isinstance(n, str) and n in library.countermeasures]
        self.log.append({"finding": finding.mode, "proposed": out.get("countermeasures"), "kept": names})
        return names[: self.max_proposals]

    def propose_concept(self, text: str, requirements: list[Requirement], name: str = "concept",
                        library: Library = LIBRARY) -> Concept:
        needs = {n.name: f"{n.label} [{n.unit}]" for r in requirements for m in r.modes
                 for n in library.mode(m).needs}
        schema = {"params": {k: "number in SI units" for k in needs}, "elements": ["short element names"],
                  "parameter meanings": needs}
        out = self.llm.extract("Propose the simplest (naive) concept that could meet these requirements: "
                               + json.dumps([text_of(r.text, "en") for r in requirements]) + " Context: " + text,
                               schema)
        params = {k: float(v) for k, v in (out.get("params") or {}).items()
                  if k in needs and isinstance(v, (int, float))}
        self.log.append({"concept": out, "kept": sorted(params)})
        return Concept(name, params, [str(e) for e in (out.get("elements") or [])][:8], {"en": text, "pt": text},
                       {k: "llm proposal (premise)" for k in params}, 0)
