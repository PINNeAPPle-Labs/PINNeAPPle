"""``ModelCard``: the reproducibility contract every hub-pushed model
carries.

Why this exists: a physical claim ("matches DNS to 2%") is only as good
as its provenance. Without architecture + exact training config + data
lineage + a computed (not just asserted) validation-metrics block bound to
the weights, a downloaded checkpoint's claims cannot be re-checked by
anyone but the person who trained it -- exactly the kind of unverifiable
claim ``pinneapple_llm``'s ``PhysicsGuardrail`` (see that module) exists to
catch. ``ModelCard`` is the structured version of what this very project's
own ``problem_config.json`` recorded by hand (physics/domain/BC/IC/
forcing/data/training/reference sections) for a single case; this is the
general, hub-wide schema.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelCard:
    name: str
    description: str = ""

    # -- what the model does and where it may be used (model-zoo fields; optional for any other card) ---------
    task: str = ""
    version: str = ""
    inputs: list[dict[str, Any]] = field(default_factory=list)    # [{"name", "unit", "description"}]
    outputs: list[dict[str, Any]] = field(default_factory=list)
    validity_range: dict[str, Any] = field(default_factory=dict)  # input -> [lo, hi], plus regime notes
    known_limitations: list[str] = field(default_factory=list)

    # -- architecture --------------------------------------------------
    architecture: str = ""  # ModelRegistry name, e.g. "modified_mlp"
    architecture_config: dict[str, Any] = field(default_factory=dict)  # in_dim/out_dim/hidden_dim/...

    # -- physics --------------------------------------------------------
    equations: list[str] = field(default_factory=list)
    domain_bounds: dict[str, Any] = field(default_factory=dict)
    boundary_conditions: dict[str, Any] = field(default_factory=dict)
    initial_condition: dict[str, Any] = field(default_factory=dict)

    # -- training ---------------------------------------------------------
    training_config: dict[str, Any] = field(default_factory=dict)

    # -- data lineage -----------------------------------------------------
    data_lineage: dict[str, Any] = field(default_factory=dict)  # source file(s)/dataset(s), hashes, time window

    # -- cost: what it took, measured (#40) --------------------------------
    cost: dict[str, Any] = field(default_factory=dict)  # wall_time_s, trainable_parameters, epochs, ...

    # -- validation (the part that has to be a computed number, not prose) --
    validation_metrics: dict[str, float] = field(default_factory=dict)
    reference_source: str = ""  # citation for whatever the validation_metrics were computed against

    # -- provenance (issue #39): what computed the numbers, with which code, seed and inputs --
    # the fields and the rule live in pinneapple_data.provenance; a card without them cannot be re-derived
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- misc ---------------------------------------------------------------
    citation: str = ""
    license: str = "apache-2.0"
    tags: list[str] = field(default_factory=list)
    pinneapple_version: str = ""

    # -- trust (decision D2) ----------------------------------------------------
    # latest TrustReport (``TrustReport.to_dict()``) and, if a REJECT is published anyway, the recorded override
    trust_report: dict[str, Any] = field(default_factory=dict)
    trust_override: dict[str, Any] = field(default_factory=dict)

    def attach_trust_report(self, report: Any) -> None:
        """Store the latest TrustReport (a ``TrustReport`` or its ``to_dict()``); a new report clears an old override."""
        self.trust_report = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        self.trust_override = {}

    def override_trust(self, reason: str, by: str = "") -> None:
        """Allow publication despite a REJECT; the reason is recorded in the card."""
        if not reason or not reason.strip():
            raise ValueError("an override needs a reason")
        from datetime import datetime, timezone
        self.trust_override = {"reason": reason.strip(), "by": by,
                               "decision": self.trust_report.get("decisao", ""),
                               "at": datetime.now(timezone.utc).isoformat()}

    @property
    def trust_decision(self) -> str:
        return self.trust_report.get("decisao", "")

    def validate(self, *, require_cost: bool = False, require_validity: bool = False) -> list[str]:
        """Return a list of schema problems (empty = passes). Does not
        raise -- callers (e.g. a CI governance check, see
        ``scripts/validate_model_card.py``) decide whether to treat a
        non-empty list as fatal.

        ``require_cost`` and ``require_validity`` are what a *generated* card has to satisfy (#40): the cost of
        producing the answer and the domain it is valid in are part of the claim, not decoration, so the generator
        asks for them explicitly and a card that omits them is refused."""
        problems = []
        if not self.name:
            problems.append("name is required")
        if not self.architecture:
            problems.append("architecture is required (a pinneapple_neural.architectures.ModelRegistry name)")
        if not self.validation_metrics:
            problems.append(
                "validation_metrics is empty -- a model card with no computed validation metric makes no "
                "checkable physical claim at all"
            )
        if require_validity and not self.validity_range:
            problems.append("validity_range is empty -- a card that does not say where the model is valid invites "
                            "its use everywhere")
        if require_cost and not self.cost:
            problems.append("cost is empty -- how long and how much memory it took is part of what a reader needs")
        if require_cost and self.cost and "wall_time_s" not in self.cost:
            problems.append("cost has no wall_time_s: the measured time is the number that makes the rest of the "
                            "cost record checkable")
        if self.validation_metrics and not self.reference_source:
            problems.append("validation_metrics is set but reference_source (what it was computed against) is empty")
        # Provenance (#39): a card whose numbers cannot be re-derived is a claim, not a record. The rule -- which
        # fields, and why each one -- is pinneapple_data.provenance; the fallback keeps this module importable
        # without pinneapple_data (the hub is usable on its own).
        try:
            from pinneapple_data.provenance import problems as provenance_problems
        except Exception:  # pragma: no cover - pinneapple_data is a hard dependency of the repo, soft of the hub
            provenance_problems = None
        if provenance_problems is not None:
            provenance = dict(self.provenance or {})
            provenance.setdefault("library_version", self.pinneapple_version)
            problems.extend(f"provenance: {line}" for line in provenance_problems(provenance))
        if self.trust_decision == "REJECT" and not self.trust_override.get("reason"):
            reasons = "; ".join(self.trust_report.get("motivos", [])) or "no reasons recorded"
            problems.append(f"trust report decision is REJECT ({reasons}) -- publication is blocked unless "
                            "overridden with a recorded reason (ModelCard.override_trust)")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_system(self) -> Any:
        """The physics ontology view of this card (#204).

        ``inputs``/``outputs`` become fields whose unit is *parsed* (a card that says ``unit="furlongs"`` fails here
        instead of being published), ``equations`` become the governing equation, ``boundary_conditions`` and
        ``initial_condition`` become typed conditions, and ``architecture`` becomes the model family. A card that
        predicts a field it does not list as an output is refused, which is the kind of card that cannot be re-checked.
        """
        from pinneapple_core.ontology import PhysicalSystem

        return PhysicalSystem.from_model_card(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> ModelCard:
        with open(path, "r") as f:
            data = json.load(f)
        return ModelCard(**data)

    def _trust_markdown(self) -> list[str]:
        if not self.trust_report:
            return []
        r = self.trust_report
        score = r.get("score")
        out = ["## Trust", f"Decision: **{r.get('decisao', '')}** · score {'—' if score is None else f'{score:.2f}'}"
               f" · coverage {r.get('cobertura', 0):.0%}", ""]
        out += [f"- {m}" for m in r.get("motivos", [])]
        if self.trust_override.get("reason"):
            out += ["", f"Published despite {self.trust_override.get('decision')}: {self.trust_override['reason']}"]
        return out + [""]

    def _io_markdown(self) -> list[str]:
        out: list[str] = []
        for title, rows in (("Inputs", self.inputs), ("Outputs", self.outputs)):
            if rows:
                out += [f"## {title}", "", "| Name | Unit | Description |", "|---|---|---|",
                        *(f"| `{r.get('name', '')}` | {r.get('unit', '')} | {r.get('description', '')} |" for r in rows),
                        ""]
        if self.validity_range:
            out += ["## Validity range", "", *(f"- `{k}`: {v}" for k, v in self.validity_range.items()), ""]
        if self.known_limitations:
            out += ["## Known limitations", "", *(f"- {x}" for x in self.known_limitations), ""]
        return out

    def to_markdown(self) -> str:
        """A human-readable card, in the same spirit as a Hugging Face
        Hub README/model card (this is uploaded as the repo's README.md
        by :func:`pinneapple_hub.hub.push_to_hub`)."""
        lines = [
            f"# {self.name}",
            "",
            self.description,
            "",
            *([f"**Task:** {self.task}", ""] if self.task else []),
            *self._io_markdown(),
            "## Architecture",
            f"- `{self.architecture}` — `{json.dumps(self.architecture_config)}`",
            "",
            "## Physics",
            *(f"- {eq}" for eq in self.equations),
            "",
            "## Validation",
            f"Reference: {self.reference_source or '(none recorded)'}",
            "",
            "| Metric | Value |",
            "|---|---|",
            *(f"| {k} | {v} |" for k, v in self.validation_metrics.items()),
            "",
            *(["## Cost", *(f"- {k}: {v}" for k, v in self.cost.items()), ""] if self.cost else []),
            *self._trust_markdown(),
            *(["## Training data", "", *(f"- {k}: {v}" for k, v in self.data_lineage.items()), ""]
              if self.data_lineage else []),
            f"License: {self.license}",
            "",
            "## Citation",
            f"```\n{self.citation}\n```" if self.citation else "(none provided)",
        ]
        return "\n".join(lines)
