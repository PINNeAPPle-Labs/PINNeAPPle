"""Explicit scientific requirements mapped to measured VeriPhysics gates."""
from dataclasses import dataclass
from hashlib import sha256
import json

@dataclass(frozen=True)
class IntentSpec:
    objective: str
    equations: tuple[str, ...]
    assumptions: tuple[str, ...]
    units: tuple[str, ...]
    required_gates: tuple[str, ...]

    def __post_init__(self):
        if not self.objective.strip() or not self.equations or not self.units or not self.required_gates:
            raise ValueError("Incomplete scientific specification")
        if len(set(self.required_gates)) != len(self.required_gates):
            raise ValueError("Duplicate requirement")

    def evaluate(self, report):
        outcomes = {}
        for name in self.required_gates:
            matches = [r for r in report.results if r.check_id == name]
            outcomes[name] = matches[0].status if len(matches) == 1 else "not run"
        status = ("failed" if "failed" in outcomes.values() else
                  "unknown" if "not run" in outcomes.values() else "passed")
        return {"intent_sha256": self.digest(), "status": status, "requirements": outcomes}

    def digest(self):
        from dataclasses import asdict
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
