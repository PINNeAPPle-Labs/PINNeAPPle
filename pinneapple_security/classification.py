"""Data classification and handling policy for datasets, models and results.

Every artifact gets a :class:`DataLabel`: confidentiality level, owner, whether it holds personal data, export-control
status (e.g. an ECCN under the US EAR, or dual-use under EU Regulation 2021/821 — the label records what the owner
declared; it does not decide classification), licence and retention. :class:`HandlingPolicy` answers "may I do X with
this artifact?" for the actions that matter in research and industry (publish, share externally, train a model on it,
send to a cloud service, export to another country) and says why not. A model trained on several inputs inherits the
strictest label (:func:`combine`).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["LEVELS", "DataLabel", "HandlingPolicy", "Decision", "combine"]

LEVELS = ("public", "internal", "confidential", "restricted")
ACTIONS = ("publish", "share_external", "train", "cloud_upload", "export")


@dataclass
class DataLabel:
    level: str = "internal"
    owner: str = ""
    personal_data: bool = False
    export_control: str = ""          # "" = not controlled; else the declared classification (e.g. "EAR99", "9E003", "dual-use")
    licence: str = ""
    retention_days: int | None = None
    allowed_countries: list[str] = field(default_factory=list)   # empty = no restriction declared
    notes: str = ""

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataLabel:
        return cls(**d)


def combine(labels: Sequence[DataLabel]) -> DataLabel:
    """Label of something derived from all ``labels``: strictest level, any personal data, every export control."""
    if not labels:
        return DataLabel()
    lvl = max((lb.level for lb in labels), key=LEVELS.index)
    ec = sorted({lb.export_control for lb in labels if lb.export_control and lb.export_control != "EAR99"})
    countries = None
    for lb in labels:
        if lb.allowed_countries:
            countries = set(lb.allowed_countries) if countries is None else countries & set(lb.allowed_countries)
    ret = [lb.retention_days for lb in labels if lb.retention_days is not None]
    return DataLabel(level=lvl, owner=", ".join(sorted({lb.owner for lb in labels if lb.owner})),
                     personal_data=any(lb.personal_data for lb in labels), export_control=", ".join(ec),
                     licence="; ".join(sorted({lb.licence for lb in labels if lb.licence})),
                     retention_days=min(ret) if ret else None, allowed_countries=sorted(countries or []),
                     notes="derived from " + str(len(labels)) + " inputs")


@dataclass
class Decision:
    allowed: bool
    reasons: list[str]
    requires: list[str] = field(default_factory=list)


@dataclass
class HandlingPolicy:
    """Default rules (organisation-specific: subclass or edit). ``approvals``: approvals the caller already holds,
    e.g. {"owner", "dpo", "export_officer"}."""
    publish_levels: Sequence[str] = ("public",)
    share_levels: Sequence[str] = ("public", "internal")
    cloud_levels: Sequence[str] = ("public", "internal", "confidential")

    def check(self, action: str, label: DataLabel, approvals: Sequence[str] = (), destination_country: str = "",
              anonymized: bool = False) -> Decision:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        approvals = set(approvals)
        reasons, requires = [], []
        if action == "publish" and label.level not in self.publish_levels:
            reasons.append(f"level '{label.level}' can not be published")
        if action == "share_external" and label.level not in self.share_levels and "owner" not in approvals:
            reasons.append(f"level '{label.level}' needs the owner's approval to share externally")
            requires.append("owner")
        if action == "cloud_upload" and label.level not in self.cloud_levels:
            reasons.append(f"level '{label.level}' must stay on premises")
        if label.personal_data and action in ("publish", "share_external", "cloud_upload", "export") and not anonymized \
                and "dpo" not in approvals:
            reasons.append("contains personal data: anonymise it or get the data-protection officer's approval (LGPD/GDPR)")
            requires.append("dpo")
        if label.export_control and label.export_control != "EAR99" and action in ("publish", "share_external", "export", "cloud_upload") \
                and "export_officer" not in approvals:
            reasons.append(f"export-controlled ({label.export_control}): needs the export-control officer's approval")
            requires.append("export_officer")
        if action == "export" and destination_country and label.allowed_countries \
                and destination_country.upper() not in {c.upper() for c in label.allowed_countries}:
            reasons.append(f"destination {destination_country} not in the allowed countries {label.allowed_countries}")
        if action == "train" and label.licence and any(x in label.licence.lower() for x in ("nc", "noncommercial", "no-ai", "noai")):
            reasons.append(f"licence '{label.licence}' may forbid this use: check it before training a commercial model")
            requires.append("legal")
        return Decision(allowed=not reasons, reasons=reasons, requires=sorted(set(requires)))
