"""Versioned, strict campaign configuration for Large Physics Models."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


class CampaignValidationError(ValueError):
    """Invalid or unsupported LPM campaign."""


_SUPPORTED = {
    ("cfd", "openfoam"): "cavity2d",
    ("structural", "calculix"): "cantilever3d",
    ("structural", "fenics"): "cantilever3d",
}


@dataclass
class Campaign:
    name: str
    domain: str
    solver: str
    family: str
    samples: int = 16
    seed: int = 42
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    geometry: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    numerical: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Campaign":
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise CampaignValidationError("Campaign config must be JSON or YAML")
        if not isinstance(raw, Mapping):
            raise CampaignValidationError("Campaign config must be an object")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise CampaignValidationError(f"Unknown campaign fields: {sorted(unknown)}")
        try:
            campaign = cls(**raw)
        except TypeError as exc:
            raise CampaignValidationError(str(exc)) from exc
        campaign.validate()
        return campaign

    def validate(self) -> None:
        expected = _SUPPORTED.get((self.domain, self.solver))
        if expected is None:
            raise CampaignValidationError(
                f"Unsupported domain/solver: {self.domain}/{self.solver}"
            )
        if self.family != expected:
            raise CampaignValidationError(
                f"Family {self.family!r} incompatible with {self.domain}/{self.solver}; expected {expected}"
            )
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 3:
            raise CampaignValidationError("samples must be an integer >= 3")
        if len(self.split) != 3 or any(
            not isinstance(x, (int, float)) or x <= 0 for x in self.split
        ) or abs(sum(self.split) - 1) > 1e-8:
            raise CampaignValidationError("split must contain three positive fractions summing to 1")
        for group in (self.geometry, self.parameters):
            for name, bounds in group.items():
                if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                    raise CampaignValidationError(f"{name}: expected [minimum, maximum]")
                lo, hi = bounds
                if not all(isinstance(v, (int, float)) for v in bounds) or hi < lo:
                    raise CampaignValidationError(f"{name}: invalid bounds")
        dimensions = ("nx", "ny") if self.domain == "cfd" else ("nx", "ny", "nz")
        if any(not isinstance(self.numerical.get(k), int) or self.numerical[k] < 2 for k in dimensions):
            raise CampaignValidationError(f"numerical must specify positive grid dimensions {dimensions}")

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}
