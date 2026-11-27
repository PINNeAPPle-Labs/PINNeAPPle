"""The intent contract: one typed, versioned description of a physics study, with the provenance of every value.

A :class:`StudyContract` is what the compiler produces from a sentence and what the runner consumes. It is the
intermediate representation the whole capability is built on, and it has three properties that matter:

* **It is self-contained.** Every value carries its SI quantity, its unit and where it came from (``stated``,
  ``answered``, ``defaulted`` with a source, ``missing``). A value nobody gave and no reference fixes is *not*
  invented: it becomes a question. The contract is therefore reconstructible without the agent that wrote it.
* **It is serialisable and addressable.** ``to_dict()`` is the whole document, ``id`` is its SHA-256, so a run, a
  report and a contract can be tied together by hash and replayed later.
* **It reuses the existing value vocabulary.** :class:`~pinneapple_design.intent.spec.Quantity`,
  :class:`~pinneapple_design.intent.spec.Slot`, :class:`~pinneapple_design.intent.spec.Question` and the
  provenance constants are the ones ``pp.design`` already uses, so there is one notion of "a number that knows where
  it came from" in the library and not two (see ``docs/strategy/STRATEGY_PHYSICS_AI.md``, decision D1).

The contract does not execute anything. :mod:`pinneapple_intent.execute` runs it; :mod:`pinneapple_intent.compile`
writes it.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pinneapple_design.intent.spec import (
    ANSWERED,
    DEFAULTED,
    DERIVED,
    MISSING,
    STATED,
    Quantity,
    Question,
    Slot,
    SpecError,
)

SCHEMA_ID = "pinneapple.study/1"
RUN_SCHEMA_ID = "pinneapple.study.run/1"

# The three ways into the pipeline. ``simulation`` and ``physics_ai`` execute end to end today; ``science`` (a
# hypothesis, an article, a benchmark reproduction) needs a literature stage that is not in this slice, so the
# compiler refuses it with what it would need instead of pretending.
MODE_SIMULATION = "simulation"
MODE_PHYSICS_AI = "physics_ai"
MODE_SCIENCE = "science"
MODES = (MODE_SIMULATION, MODE_PHYSICS_AI, MODE_SCIENCE)
EXECUTABLE_MODES = (MODE_SIMULATION, MODE_PHYSICS_AI)

READY = "ready"
NEEDS_CLARIFICATION = "needs_clarification"
OUT_OF_SCOPE = "out_of_scope"
STATUSES = (READY, NEEDS_CLARIFICATION, OUT_OF_SCOPE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(obj: Any) -> str:
    """The JSON text an id is computed from: sorted keys, no whitespace, stable floats."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class Provenance:
    """Where the contract came from: the sentence, the compiler that read it, the library that will run it.

    ``compiler`` is ``deterministic`` or ``llm+deterministic``; ``llm`` names the model when one was asked, and its
    answer is kept here even when the deterministic parser overrode it, because "the LLM proposed 85 degC and the
    text did not contain it" is part of the record.
    """
    text: str = ""
    compiler: str = "deterministic"
    llm: str = ""
    llm_note: str = ""
    library_version: str = ""
    created_at: str = field(default_factory=_now)
    seed: int = 0

    @classmethod
    def capture(cls, text: str, *, compiler: str = "deterministic", llm: str = "", llm_note: str = "",
                seed: int = 0) -> Provenance:
        try:
            from pinneapple import __version__ as version
        except Exception:  # noqa: BLE001 - a contract is still valid without the version string
            version = ""
        return cls(text=text, compiler=compiler, llm=llm, llm_note=llm_note, library_version=version, seed=seed)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "compiler": self.compiler, "llm": self.llm, "llm_note": self.llm_note,
                "library_version": self.library_version, "created_at": self.created_at, "seed": self.seed,
                "python": platform.python_version(), "platform": platform.system()}


@dataclass(frozen=True)
class Rung:
    """One step of the fidelity ladder: what is solved, how, and what it needs to be installed."""
    name: str                              # analytic | numerical | convergence | surrogate
    label: str                             # "1-D series resistance", "immersed finite volume"
    method: str                            # one line: what is solved and how
    modules: tuple[str, ...] = ()
    executables: tuple[str, ...] = ()

    def availability(self) -> tuple[bool, str]:
        import importlib.util
        import shutil
        for module in self.modules:
            if importlib.util.find_spec(module) is None:
                return False, f"Python module '{module}' is not installed"
        for exe in self.executables:
            if not any(shutil.which(e) for e in exe.split("|")):
                return False, f"program '{exe}' is not on PATH"
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        ok, why = self.availability()
        return {"name": self.name, "label": self.label, "method": self.method, "available": ok, "reason": why}


@dataclass
class StudyContract:
    """A physics study, described well enough to be run again by someone who never saw the request.

    ``family`` names the registered :class:`~pinneapple_intent.families.Family` that supplies the equations, the
    quantities it needs and the ladder. ``values`` are the filled slots. ``assumptions`` are the statements the
    family makes that are *not* measured (a reported study lists them); ``limitations`` are what the study does not
    cover, so a reader cannot take the result further than it goes.
    """
    family: str
    mode: str = MODE_SIMULATION
    objective: str = ""
    title: str = ""
    domain: str = ""
    equations: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    values: dict[str, Quantity] = field(default_factory=dict)
    ladder: tuple[Rung, ...] = ()
    validation: tuple[str, ...] = ()
    questions: list[Question] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)

    # ------------------------------------------------------------------ status and identity
    @property
    def status(self) -> str:
        if self.questions or self.issues:
            return NEEDS_CLARIFICATION
        return READY

    @property
    def id(self) -> str:
        """The SHA-256 of the document without its creation timestamp, so the same contract has the same id."""
        doc = self.to_dict()
        doc.pop("id", None)
        doc["provenance"].pop("created_at", None)
        return digest(doc)

    @property
    def rung_names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.ladder)

    def __getitem__(self, name: str) -> Any:
        return self.values[name].value

    def get(self, name: str, default: Any = None) -> Any:
        q = self.values.get(name)
        return q.value if q is not None and q.known else default

    def quantity(self, name: str) -> Quantity:
        return self.values[name]

    def by_provenance(self, provenance: str) -> list[Quantity]:
        return [q for q in self.values.values() if q.provenance == provenance]

    # ------------------------------------------------------------------ with / without an answer
    def answer(self, slots: dict[str, Slot], **answers: Any) -> StudyContract:
        """A new contract with the follow-up answers filled in. ``name=(value, unit)``, ``name="70 degC"`` or
        ``name=value`` in the unit the question showed."""
        values = dict(self.values)
        issues = list(self.issues)
        for name, raw in answers.items():
            if name not in slots:
                raise SpecError(f"unknown value {name!r}; this study takes {sorted(slots)}")
            slot = slots[name]
            value, unit = _split_answer(slot, raw)
            values[name] = slot.quantity(value, unit, ANSWERED, source="user (follow-up)")
            issues = [i for i in issues if not i.startswith(slot.label)]
        questions = [q for q in self.questions if q.slot not in answers]
        return StudyContract(**{**self.__dict__, "values": values, "issues": issues, "questions": questions,
                                "provenance": self.provenance})

    def with_mode(self, mode: str) -> StudyContract:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; known: {list(MODES)}")
        return StudyContract(**{**self.__dict__, "mode": mode})

    # ------------------------------------------------------------------ documents
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_ID,
            "id": None,
            "status": self.status,
            "mode": self.mode,
            "family": self.family,
            "title": self.title,
            "objective": self.objective,
            "domain": self.domain,
            "physics": {"equations": list(self.equations), "assumptions": list(self.assumptions)},
            "limitations": list(self.limitations),
            "references": list(self.references),
            "quantities": {k: v.to_dict() for k, v in self.values.items()},
            "workflow": {"ladder": [r.to_dict() for r in self.ladder], "validation": list(self.validation)},
            "questions": [q.to_dict() for q in self.questions],
            "issues": list(self.issues),
            "unsupported": list(self.unsupported),
            "provenance": self.provenance.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        doc = dict(self.to_dict())
        doc["id"] = self.id
        return json.dumps(doc, indent=indent, sort_keys=False, default=str)

    def write(self, path: str) -> str:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
        return path

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> StudyContract:
        if doc.get("schema") not in (SCHEMA_ID, None):
            raise ValueError(f"not a {SCHEMA_ID} document: {doc.get('schema')!r}")
        physics = doc.get("physics") or {}
        workflow = doc.get("workflow") or {}
        prov = doc.get("provenance") or {}
        values = {k: Quantity(**v) for k, v in (doc.get("quantities") or {}).items()}
        return cls(
            family=doc["family"], mode=doc.get("mode", MODE_SIMULATION), objective=doc.get("objective", ""),
            title=doc.get("title", ""), domain=doc.get("domain", ""),
            equations=tuple(physics.get("equations") or ()), assumptions=tuple(physics.get("assumptions") or ()),
            limitations=tuple(doc.get("limitations") or ()), references=tuple(doc.get("references") or ()),
            values=values,
            ladder=tuple(Rung(name=r["name"], label=r["label"], method=r["method"], modules=tuple(r.get("modules") or ()))
                         for r in workflow.get("ladder") or ()),
            validation=tuple(workflow.get("validation") or ()),
            questions=[Question(**q) for q in doc.get("questions") or ()],
            issues=list(doc.get("issues") or ()), unsupported=list(doc.get("unsupported") or ()),
            provenance=Provenance(text=prov.get("text", ""), compiler=prov.get("compiler", "deterministic"),
                                  llm=prov.get("llm", ""), llm_note=prov.get("llm_note", ""),
                                  library_version=prov.get("library_version", ""),
                                  created_at=prov.get("created_at", ""), seed=int(prov.get("seed", 0) or 0)),
        )

    @classmethod
    def read(cls, path: str) -> StudyContract:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # ------------------------------------------------------------------ prose
    def summary(self) -> str:
        if self.status == NEEDS_CLARIFICATION:
            pending = "; ".join(f"{q.text}" + (f" [{q.unit}]" if q.unit else "") for q in self.questions)
            return f"{self.title or self.family}: needs input ({pending or '; '.join(self.issues)})"
        given = ", ".join(f"{v.display()}" for v in self.values.values() if v.known)
        return f"{self.title or self.family} ({self.family}, {self.mode}): {given}"

    def table(self) -> list[dict[str, Any]]:
        """One row per quantity, with the provenance a report has to show next to every number."""
        rows = []
        for quantity in self.values.values():
            rows.append({"name": quantity.name, "value": quantity.display(), "provenance": quantity.provenance,
                         "source": quantity.quote or quantity.source})
        return rows


def _split_answer(slot: Slot, raw: Any) -> tuple[Any, str]:
    """Read one follow-up answer: a tuple, a "70 degC" string, or a bare number in the slot's shown unit."""
    import re
    if isinstance(raw, tuple):
        return raw[0], raw[1]
    if isinstance(raw, str) and not slot.is_choice:
        found = re.fullmatch(r"\s*([-+]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*", raw)
        if not found:
            raise SpecError(f"{slot.label}: cannot read {raw!r}; give a number and a unit, e.g. "
                            f"'70 {slot.shown_unit()}'")
        return float(found.group(1).replace(",", ".")), found.group(2) or slot.shown_unit()
    return raw, ("" if slot.is_choice else slot.shown_unit())


__all__ = [
    "SCHEMA_ID", "RUN_SCHEMA_ID", "MODES", "EXECUTABLE_MODES", "MODE_SIMULATION", "MODE_PHYSICS_AI", "MODE_SCIENCE",
    "READY", "NEEDS_CLARIFICATION", "OUT_OF_SCOPE", "STATUSES",
    "Provenance", "Rung", "StudyContract",
    "Quantity", "Slot", "Question", "SpecError", "STATED", "ANSWERED", "DEFAULTED", "MISSING", "DERIVED",
    "canonical", "digest",
]
