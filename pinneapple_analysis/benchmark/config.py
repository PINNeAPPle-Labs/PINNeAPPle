"""Benchmark configuration: the same fields from Python keyword arguments or a YAML file, resolved and hashed."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_core.errors import BadValue

__all__ = ["BenchmarkConfig"]

# fields that change what a job computes: a cache built with other values is refused unless forced
_RESULT_FIELDS = ("problems", "methods", "budget", "problem_budgets", "budget_levels", "seeds", "n_trials",
                  "search_fraction", "equal_flops", "threads_per_run", "smoke", "problem_options", "method_options")


def _budget_dict(b) -> dict[str, Any]:
    if b is None:
        return {}
    if isinstance(b, ComputeBudget):
        return {k: v for k, v in b.to_dict().items() if v is not None}
    if isinstance(b, (int, float)):
        return {"wall_s": float(b)}
    return {k: v for k, v in dict(b).items() if v is not None}


@dataclass
class BenchmarkConfig:
    """Everything a benchmark run depends on.

    ``problems`` / ``methods``: registered ids (``pp.benchmark.problems()``, ``methods()``) or objects.
    ``budget``: the full training budget of one run (``ComputeBudget`` or a dict ``{wall_s|flops|steps: ...}``);
    ``problem_budgets`` overrides it per problem. ``budget_levels``: fractions of it trained for the error-versus-budget
    curves (the largest must be 1). ``seeds``: number of seeds (0..seeds-1). ``n_trials`` configurations per method in
    the search, each trained for ``search_fraction`` of the budget on validation data only. ``equal_flops``: True for
    every problem, False, or a list of problem ids. ``threads_per_run`` threads per training process, ``n_workers``
    processes at a time. ``smoke``: tiny data and budgets, a pipeline check in minutes. ``skip_checks`` /
    ``allow_failed_validation``: run anyway; the report then says so on every page.
    ``budget_study``: ``{problem: {method: [wall-clock seconds, ...]}}``, extra runs of the best configuration (seed 0,
    the problem's study instances) at budgets beyond the full one: is a method budget-limited? ``ablations``:
    ``{problem: {method: {name: {hyper-parameter: value}}}}``, the best configuration with these changes at the full
    budget on the problem's audit instance, compared with the unchanged run (a setup audit).
    """
    problems: list[Any] = field(default_factory=lambda: ["burgers"])
    methods: list[Any] = field(default_factory=lambda: ["pinn", "fno", "deeponet"])
    budget: dict[str, Any] = field(default_factory=lambda: {"wall_s": 300.0})
    problem_budgets: dict[str, dict[str, Any]] = field(default_factory=dict)
    budget_levels: list[float] = field(default_factory=lambda: [1.0])
    seeds: int = 3
    n_trials: int = 4
    search_fraction: float = 1 / 3
    equal_flops: Any = False
    n_workers: int = 1
    threads_per_run: int = 1
    smoke: bool = False
    skip_checks: bool = False
    allow_failed_validation: bool = False
    problem_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    method_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    budget_study: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    ablations: dict[str, dict[str, dict[str, dict[str, Any]]]] = field(default_factory=dict)

    def __post_init__(self):
        self.budget = _budget_dict(self.budget)
        if not self.budget:
            raise BadValue("a benchmark run needs a compute budget: it is what makes two methods comparable",
                           fix="pass budget={'wall_s': 60} (or flops=... / steps=...)")
        self.problem_budgets = {k: _budget_dict(v) for k, v in (self.problem_budgets or {}).items()}
        self.budget_levels = sorted(float(x) for x in (self.budget_levels or [1.0]))
        if abs(self.budget_levels[-1] - 1.0) > 1e-9 or self.budget_levels[0] <= 0:
            raise BadValue(f"budget_levels must be fractions in (0, 1] ending at 1.0, got {self.budget_levels}",
                           fix="use e.g. [0.25, 0.5, 1.0]: each level is a fraction of the budget, and the last "
                               "one has to be the full budget")
        if self.seeds < 1 or self.n_trials < 1 or not 0 < self.search_fraction <= 1:
            raise BadValue(f"invalid run shape: seeds={self.seeds}, n_trials={self.n_trials}, "
                           f"search_fraction={self.search_fraction}",
                           fix="use seeds >= 1, n_trials >= 1 and 0 < search_fraction <= 1 (1/3 is the default)")
        if isinstance(self.problems, (str, bytes)) or not isinstance(self.problems, (list, tuple)):
            self.problems = [self.problems]
        if isinstance(self.methods, (str, bytes)) or not isinstance(self.methods, (list, tuple)):
            self.methods = [self.methods]

    # ------------------------------------------------------------------ ids and budgets
    @staticmethod
    def _id(x) -> str:
        return x if isinstance(x, str) else getattr(x, "id", type(x).__name__)

    @property
    def problem_ids(self) -> list[str]:
        return [self._id(p) for p in self.problems]

    @property
    def method_ids(self) -> list[str]:
        return [self._id(m) for m in self.methods]

    def budget_for(self, problem_id: str, smoke: bool | None = None) -> ComputeBudget:
        b = dict(self.problem_budgets.get(problem_id, self.budget))
        if self.smoke if smoke is None else smoke:
            b = {k: (min(v, 6.0) if k == "wall_s" else max(1, int(v * 0.01)) if k == "steps" else v * 0.01)
                 for k, v in b.items()}
        return ComputeBudget(**b)

    def equal_flops_for(self, problem_id: str) -> bool:
        if isinstance(self.equal_flops, bool):
            return self.equal_flops
        return problem_id in set(self.equal_flops or [])

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["problems"], d["methods"] = self.problem_ids, self.method_ids
        return d

    def result_hash(self) -> str:
        d = self.to_dict()
        key = {k: d[k] for k in _RESULT_FIELDS}       # budget_study / ablations only add jobs: see extends_studies
        return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def extends_studies(self, old: BenchmarkConfig) -> bool:
        """True when this configuration's ``budget_study`` and ``ablations`` keep every entry of ``old`` unchanged
        (they may add problems, methods, budgets or variants: new jobs, no cached result reinterpreted)."""
        for pid, per in (old.budget_study or {}).items():
            for mid, bs in per.items():
                new = (self.budget_study.get(pid) or {}).get(mid, [])
                if not {float(x) for x in bs} <= {float(x) for x in new}:
                    return False
        for pid, per in (old.ablations or {}).items():
            for mid, variants in per.items():
                new = (self.ablations.get(pid) or {}).get(mid, {})
                if any(new.get(name) != change for name, change in variants.items()):
                    return False
        return True

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkConfig:
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise BadValue(f"unknown benchmark config field(s) {sorted(unknown)}",
                           fix=f"drop or rename them; the known fields are {sorted(names)}")
        return cls(**d)
