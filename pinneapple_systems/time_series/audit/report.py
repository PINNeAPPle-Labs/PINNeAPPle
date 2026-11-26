from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditSection:
    name: str
    results: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "results": self.results}


@dataclass
class AuditReport:
    """Serializable container for time-series diagnostic outputs."""
    meta: dict[str, Any] = field(default_factory=dict)
    sections: dict[str, AuditSection] = field(default_factory=dict)

    def add(self, section: str, key: str, value: Any) -> None:
        if section not in self.sections:
            self.sections[section] = AuditSection(name=section)
        self.sections[section].results[key] = value

    def to_dict(self) -> dict[str, Any]:
        return {"meta": self.meta, "sections": {k: v.to_dict() for k, v in self.sections.items()}}