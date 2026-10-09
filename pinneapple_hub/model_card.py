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
from typing import Any, Dict, List, Optional


@dataclass
class ModelCard:
    name: str
    description: str = ""

    # -- architecture --------------------------------------------------
    architecture: str = ""  # ModelRegistry name, e.g. "modified_mlp"
    architecture_config: Dict[str, Any] = field(default_factory=dict)  # in_dim/out_dim/hidden_dim/...

    # -- physics --------------------------------------------------------
    equations: List[str] = field(default_factory=list)
    domain_bounds: Dict[str, Any] = field(default_factory=dict)
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    initial_condition: Dict[str, Any] = field(default_factory=dict)

    # -- training ---------------------------------------------------------
    training_config: Dict[str, Any] = field(default_factory=dict)

    # -- data lineage -----------------------------------------------------
    data_lineage: Dict[str, Any] = field(default_factory=dict)  # source file(s)/dataset(s), hashes, time window

    # -- validation (the part that has to be a computed number, not prose) --
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    reference_source: str = ""  # citation for whatever the validation_metrics were computed against

    # -- misc ---------------------------------------------------------------
    citation: str = ""
    license: str = "apache-2.0"
    tags: List[str] = field(default_factory=list)
    pinneapple_version: str = ""

    # -- trust (decision D2) ----------------------------------------------------
    # latest TrustReport (``TrustReport.to_dict()``) and, if a REJECT is published anyway, the recorded override
    trust_report: Dict[str, Any] = field(default_factory=dict)
    trust_override: Dict[str, Any] = field(default_factory=dict)

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

    def validate(self) -> List[str]:
        """Return a list of schema problems (empty = passes). Does not
        raise -- callers (e.g. a CI governance check, see
        ``scripts/validate_model_card.py``) decide whether to treat a
        non-empty list as fatal."""
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
        if self.validation_metrics and not self.reference_source:
            problems.append("validation_metrics is set but reference_source (what it was computed against) is empty")
        if self.trust_decision == "REJECT" and not self.trust_override.get("reason"):
            reasons = "; ".join(self.trust_report.get("motivos", [])) or "no reasons recorded"
            problems.append(f"trust report decision is REJECT ({reasons}) -- publication is blocked unless "
                            "overridden with a recorded reason (ModelCard.override_trust)")
        return problems

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "ModelCard":
        with open(path, "r") as f:
            data = json.load(f)
        return ModelCard(**data)

    def _trust_markdown(self) -> List[str]:
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

    def to_markdown(self) -> str:
        """A human-readable card, in the same spirit as a Hugging Face
        Hub README/model card (this is uploaded as the repo's README.md
        by :func:`pinneapple_hub.hub.push_to_hub`)."""
        lines = [
            f"# {self.name}",
            "",
            self.description,
            "",
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
            *self._trust_markdown(),
            "## Citation",
            f"```\n{self.citation}\n```" if self.citation else "(none provided)",
        ]
        return "\n".join(lines)
