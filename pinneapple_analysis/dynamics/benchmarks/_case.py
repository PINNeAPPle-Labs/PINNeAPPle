"""The record every benchmark returns, and its Markdown table."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkCase:
    """One comparison against a known answer.

    ``z`` is ``|error| / mc_se`` for a random method (agreement means a small ``z``, about 3 at most) and ``None``
    for a deterministic one, where ``tolerance`` applies instead. ``passed`` uses whichever applies."""
    name: str
    quantity: str
    truth: float
    truth_source: str
    value: float
    mc_se: float | None = None
    tolerance: float | None = None
    z_max: float = 4.0
    cost_s: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    convergence: dict[str, Any] = field(default_factory=dict)
    limitations: str = ""

    @property
    def error(self) -> float:
        return self.value - self.truth

    @property
    def z(self) -> float | None:
        if self.mc_se is None or not self.mc_se > 0:
            return None
        return abs(self.error) / self.mc_se

    @property
    def passed(self) -> bool:
        if self.tolerance is not None and abs(self.error) > self.tolerance:
            return False
        z = self.z
        return z is None or z <= self.z_max

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "quantity": self.quantity, "truth": self.truth, "truth_source": self.truth_source,
                "value": self.value, "error": self.error, "mc_se": self.mc_se, "z": self.z,
                "tolerance": self.tolerance, "passed": self.passed, "cost_s": self.cost_s, "config": self.config,
                "convergence": self.convergence, "limitations": self.limitations}


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        return f"{v:.4g}"
    return str(v)


def to_markdown(cases: list[BenchmarkCase]) -> str:
    """A Markdown table: case, quantity, reference, computed, error, Monte Carlo SE, z, pass, cost."""
    head = "| Case | Quantity | Reference | Computed | Error | MC s.e. | z | Pass | Cost [s] |\n|---|---|---|---|---|---|---|---|---|"
    rows = [f"| {c.name} | {c.quantity} | {_fmt(c.truth)} | {_fmt(c.value)} | {_fmt(c.error)} | {_fmt(c.mc_se)} | "
            f"{_fmt(c.z)} | {'yes' if c.passed else 'NO'} | {c.cost_s:.2f} |" for c in cases]
    return "\n".join([head, *rows])
