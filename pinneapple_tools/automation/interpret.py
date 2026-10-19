"""From a task to a structured workflow.

A task is either a workflow already (a dict, or JSON/YAML text: it is validated and used as is) or a sentence
such as "run the channel case with a 55 mm obstacle". A sentence is matched against a **catalog of committed
workflows**: the words of the task pick the workflow and the numbers next to an input's words set its value (with
unit conversion), then the inputs are checked against their bounds. Nothing in a sentence can add steps or run
code: it only chooses among workflows someone reviewed, and sets their inputs.

An LLM is optional. With ``llm=OllamaClient(...)`` (``pinneapple_data.formfill.llm``; the Ollama service of the
deploy stack) the model answers in a JSON schema with the workflow id and input values, and its answer goes
through the same validation; if the server is not reachable, or the answer is invalid, the rule-based reading is
used and the reason is kept in ``notes``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .workflow import Workflow, WorkflowError

__all__ = ["Interpretation", "interpret", "parse_quantity"]

# factor to the unit of the input: {input unit: {written unit: factor}}
_UNITS = {
    "mm": {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4},
    "m": {"mm": 1e-3, "cm": 1e-2, "m": 1.0},
    "mm/s": {"mm/s": 1.0, "cm/s": 10.0, "m/s": 1000.0},
    "m/s": {"mm/s": 1e-3, "cm/s": 1e-2, "m/s": 1.0},
    "Pa": {"pa": 1.0, "kpa": 1e3, "mbar": 100.0, "bar": 1e5},
}
_NUM = r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
_STOP = {"the", "a", "an", "of", "to", "in", "with", "and", "for", "at", "on", "by", "mm", "m", "s"}


@dataclass
class Interpretation:
    workflow: Workflow
    inputs: dict[str, Any]
    method: str                              # structured | rules | llm
    notes: list[str] = field(default_factory=list)
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"workflow": self.workflow.id, "inputs": self.inputs, "method": self.method, "notes": self.notes,
                "score": self.score}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 1}


def parse_quantity(text: str, unit: str) -> float | None:
    """The number in ``text`` converted to ``unit`` ("4 cm" -> 40 for mm); None if there is no number."""
    m = re.search(rf"({_NUM})\s*([a-zA-Z/]+)?", text)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    written = (m.group(2) or "").lower()
    table = {k.lower(): f for k, f in _UNITS.get(unit, {}).items()}
    return v * table[written] if written in table else v


def _input_words(name: str, spec) -> set[str]:
    return _words(name.replace("_", " ")) | _words(spec.description)


def _rules(task: str, catalog: dict[str, Workflow]) -> Interpretation:
    tw = _words(task)
    best, best_score = None, -1.0
    for wf in catalog.values():
        vocab = _words(f"{wf.id.replace('_', ' ')} {wf.title} {wf.description} {wf.application}")
        for n, s in wf.inputs.items():
            vocab |= _input_words(n, s)
        score = len(tw & vocab) / max(len(tw), 1)
        if score > best_score:
            best, best_score = wf, score
    if best is None:
        raise WorkflowError(["the catalog is empty"])
    notes = [f"picked {best.id!r}: {best_score:.0%} of the task's words appear in it"]
    values: dict[str, Any] = {}
    low = task.lower()
    for name, spec in best.inputs.items():
        keys = [w for w in _input_words(name, spec) if len(w) > 2]
        # a number after (or right before) one of the input's words, within a few words
        pats = [rf"\b(?:{'|'.join(map(re.escape, keys))})\w*\b(?:\W+\w+){{0,3}}?\W+({_NUM}\s*[a-zA-Z/]*)",
                rf"({_NUM}\s*[a-zA-Z/]*)\s+(?:\w+\s+){{0,1}}(?:{'|'.join(map(re.escape, keys))})"] if keys else []
        for pat in pats:
            m = re.search(pat, low)
            if m:
                q = parse_quantity(m.group(1), spec.unit)
                if q is not None:
                    values[name] = q
                    notes.append(f"{name} = {q:g} {spec.unit} from {m.group(0).strip()!r}")
                    break
    return Interpretation(best, values, "rules", notes, best_score)


def _structured(task: str | dict[str, Any]) -> Any:
    if isinstance(task, dict):
        return task
    text = task.strip()
    if text[:1] in "{[":
        return json.loads(text)
    if "\n" in text and re.match(r"^(id|steps|workflow)\s*:", text, re.M):
        import yaml
        return yaml.safe_load(text)
    return None


def interpret(task: str | dict[str, Any], catalog: dict[str, Workflow], llm=None,
              min_score: float = 0.15) -> Interpretation:
    """Turn ``task`` into a workflow of ``catalog`` and its inputs (see the module docstring)."""
    data = _structured(task)
    if data is not None:
        if isinstance(data, dict) and "steps" not in data and data.get("workflow") in catalog:
            wf = catalog[data["workflow"]]
            return Interpretation(wf, wf.resolve_inputs(data.get("inputs") or {}), "structured",
                                  ["structured request for a catalog workflow"])
        wf = Workflow.from_dict(data)
        return Interpretation(wf, wf.resolve_inputs(data.get("run_inputs") or {}), "structured",
                              ["structured workflow, validated"])
    notes: list[str] = []
    if llm is not None:
        try:
            if not llm.available():
                raise RuntimeError(f"Ollama at {llm.url} is not reachable or model {llm.model!r} is not pulled")
            schema = {"type": "object", "required": ["workflow", "inputs"],
                      "properties": {"workflow": {"type": "string", "enum": sorted(catalog)},
                                     "inputs": {"type": "object"}}}
            menu = "\n".join(f"- {w.id}: {w.title}. Inputs: " + ", ".join(
                f"{k} ({s.unit}, {s.min}..{s.max}, default {s.default}): {s.description}"
                for k, s in w.inputs.items()) for w in catalog.values())
            ans = llm.chat_json("You map an engineering task to one of the listed workflows and its input values. "
                                "Use only the listed workflow ids and input names. Leave out inputs the task does "
                                "not state. Convert values to the listed unit.", f"Workflows:\n{menu}\n\nTask: {task}",
                                schema)
            wf = catalog[ans["workflow"]]
            inputs = {k: v for k, v in (ans.get("inputs") or {}).items() if k in wf.inputs}
            wf.resolve_inputs(inputs)
            return Interpretation(wf, inputs, "llm", [f"Ollama model {llm.model} chose {wf.id}"])
        except Exception as e:                                       # fall back to the rules, say why
            notes.append(f"LLM not used: {e}")
    it = _rules(str(task), catalog)
    it.notes = notes + it.notes
    if it.score < min_score:
        raise WorkflowError([f"no committed workflow matches the task (best {it.workflow.id!r} at {it.score:.0%}); "
                             f"available: {', '.join(w.title for w in catalog.values())}"])
    try:
        it.workflow.resolve_inputs(it.inputs)
    except WorkflowError as e:
        raise WorkflowError([f"{it.workflow.id}: {x}" for x in e.errors]) from None
    return it
