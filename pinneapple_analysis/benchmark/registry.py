"""Problems and methods: base classes, the registry and the decorators that extend it.

A **Problem** owns its data (``prepare`` writes it, with its generation cost, into ``out_dir/prepared/<id>/``), the
verification of its reference solver, the data checks, the metrics (``columns``) and how a method's raw result maps to
them (``extract``). A **Method** declares its hyper-parameter grid (``space``) and trains under a ``ComputeBudget``
(``train``). A problem may carry its own implementation of a method as ``train_<method id>`` (the airfoil's
MeshGraphNet, the inverse-heat PINN); generic methods (FNO, DeepONet, PINN) work on any field-operator problem,
including a user's own arrays (``ArrayProblem``).

Register your own with the decorators::

    @pp.benchmark.register_problem("my_flow")
    class MyFlow(pp.benchmark.ArrayProblem): ...

    @pp.benchmark.register_method("my_net")
    class MyNet(pp.benchmark.Method): ...
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_core.errors import NotSupported, UnknownName

__all__ = ["Check", "Column", "Problem", "Method", "register_problem", "register_method", "get_problem",
           "get_method", "problems", "methods"]

_PROBLEMS: dict[str, Callable[[], Problem]] = {}
_METHODS: dict[str, Callable[[], Method]] = {}


@dataclass
class Check:
    """One gate or validation item. ``passed`` None means not applicable (always with a reason in ``detail``)."""
    name: str
    stage: str                       # problem | data | architecture | training | validation
    subject: str                     # "burgers", "burgers/fno", ...
    passed: bool | None
    value: Any = None
    threshold: Any = None
    detail: str = ""
    severity: str = "error"          # error: blocks / invalidates; warning: reported only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def status(self) -> str:
        return "n/a" if self.passed is None else ("pass" if self.passed else ("FAIL" if self.severity == "error" else "warn"))


@dataclass
class Column:
    """A table column: ``metric`` of the instances in ``set`` (e.g. rel_l2 on test), ``kind`` field | integral."""
    key: str
    label: str
    set: str
    metric: str
    kind: str = "field"
    pct: bool = False


class Problem:
    """Base class of a benchmark problem. Subclasses set ``id``, ``title`` and implement the hooks they need."""
    id: str = "problem"
    title: str = "Problem"
    kind: str = "field_operator"     # field_operator | geometry | inverse
    budget_note: str = ""

    def __init__(self, **options):
        self.options = {**self.default_options(), **options}

    # ------------------------------------------------------------------ configuration
    def default_options(self) -> dict[str, Any]:
        return {}

    def smoke_options(self) -> dict[str, Any]:
        return {}

    def configure(self, options: dict[str, Any] | None, smoke: bool) -> Problem:
        self.options = {**self.default_options(), **(self.smoke_options() if smoke else {}), **(options or {})}
        self.smoke = smoke
        return self

    def solver_config(self) -> dict[str, Any]:
        return dict(self.options)

    def bind(self, prepared: Path) -> Problem:
        """Tell the problem where its prepared artifacts are (some read them to list their instances)."""
        self._prepared = Path(prepared)
        return self

    def method_title(self, method) -> str:
        return getattr(self, f"title_{method.id}", method.title)

    def method_outputs(self, method) -> str:
        return getattr(self, f"outputs_{method.id}", method.outputs)

    # ------------------------------------------------------------------ data
    def prepare(self, out: Path, n_workers: int = 1) -> dict[str, Any]:
        """Generate the data into ``out`` (and geometry, if any); return a manifest with the files and their cost."""
        raise NotImplementedError

    def artifact_files(self, out: Path) -> list[str]:
        """The files ``prepare`` writes (relative to ``out``); used to adopt existing artifacts."""
        return ["cost.json"]

    def verify_solver(self) -> list[Check]:
        """Verification of the reference solver on this configuration (exact or manufactured solutions)."""
        return [Check("reference solver verification", "problem", self.id, None,
                      detail="this problem declares no verification of its reference solver")]

    def data_checks(self, out: Path) -> list[Check]:
        return []

    def validation(self, out: Path) -> list[Check]:
        """PINNeAPPle validation-stack items that apply to this problem (mesh quality, GCI, preflight...)."""
        return []

    # ------------------------------------------------------------------ evaluation
    def columns(self) -> list[Column]:
        return []

    def instances(self) -> dict[str, list[tuple[str, Any]]]:
        """For per-instance methods (PINNs), lists of (job-id suffix, keyword arguments of the trainer):
        "val" (the search instance), "final" (every budget level), "full_only" (full budget only), "seed0_only"."""
        return {"val": [], "final": [], "full_only": [], "seed0_only": []}

    def validity_domain(self, prepared: Path | None = None):
        """Where this case holds (#58): a ``validity.ValidityDomain``, or ``None`` if the case has not declared one.

        ``None`` is printed as "not stated" in the report, not silently omitted: a case that does not say where it
        holds has not said anything about extrapolation either.
        """
        return None

    def extrapolation(self) -> tuple[tuple[str, str], tuple[str, str], float] | None:
        """``((inside set, metric), (outside set, metric), max factor)`` for the extrapolation column (#58).

        The default pairs the case's own in-domain and out-of-domain sets, which every benchmark problem evaluates
        anyway; ``extrapolation_factor`` sets how much worse the out-of-domain error may be before the column says the
        model does not hold there.
        """
        main = self.main_metric() if hasattr(self, "main_metric") else None
        ood = self.ood_metric() if hasattr(self, "ood_metric") else None
        return (main, ood, self.extrapolation_factor) if main and ood else None

    extrapolation_factor: float = 2.0

    def main_metric_for(self, method_id: str) -> tuple[str, str] | None:
        """The (set, metric) that decides "did not converge", the seed spread and "budget-limited" for one method
        (default: the problem's ``main_metric``; a problem whose methods predict different things overrides it)."""
        return self.main_metric() if hasattr(self, "main_metric") else None

    def study_instances(self) -> list[tuple[str, Any]]:
        """Instances of the budget study and of the ablations for per-instance methods (default: the first "final"
        instance)."""
        return list(self.instances().get("final", []))[:1]

    def audit_instance(self) -> tuple[str, Any] | None:
        """The instance of the ablations (setup audit) for per-instance methods (default: the first study instance)."""
        inst = self.study_instances()
        return inst[0] if inst else None

    def report_sections(self, plan) -> list[str]:
        """Extra Markdown lines for this problem's section of the report (problem-specific evidence)."""
        return []

    def run_metrics(self, method_id: str, suffix: str, result: dict) -> dict[str, float]:
        """One run's metrics, flat ({"set/metric": mean}), from ``extract`` (budget study and ablation tables)."""
        out = {}
        for set_, ms in self.extract(method_id, {suffix: result}).items():
            for m, e in ms.items():
                v = [float(x) for x in e.get("values", []) if x is not None]
                if v:
                    out[f"{set_}/{m}"] = float(np.mean(v))
        return out

    def extract(self, method_id: str, results: dict[str, dict]) -> dict[str, dict[str, Any]]:
        """Map raw results to {set: {metric: {"values": [...], "ids": [...]}}}; ``results`` maps the instance suffix
        ("" for a method trained once) to that job's raw result."""
        raise NotImplementedError

    def search_score(self, method_id: str, result: dict) -> float:
        raise NotImplementedError

    def data_cost_s(self, out: Path) -> float | None:
        return None

    def solver_time_per_instance_s(self, out: Path) -> float | None:
        return None


class Method:
    """Base class of a benchmark method."""
    id: str = "method"
    title: str = "Method"
    per_instance: bool = False       # trained again for every new instance (a PINN), or once on data
    outputs: str = "fields"          # fields | integrals | both

    def supports(self, problem: Problem) -> bool:
        return hasattr(problem, f"train_{self.id}")

    def space(self, problem: Problem) -> dict[str, list]:
        sp = getattr(problem, f"space_{self.id}", None)
        if sp is None:
            raise NotSupported(f"{self.id} declares no hyper-parameter grid for {problem.id}",
                               fix=f"train it without a search (no search over space), or add space_{self.id}() "
                                   f"to {type(problem).__name__}")
        return sp() if callable(sp) else sp

    def train(self, problem: Problem, prepared: Path, cfg: dict, seed: int, budget, **job) -> dict:
        hook = getattr(problem, f"train_{self.id}", None)
        if hook is None:
            raise NotSupported(f"{self.id} has no implementation on {problem.id}",
                               fix=f"pick a pair the problem implements (see problem.pairs()), or add "
                                   f"train_{self.id}() to {type(problem).__name__}")
        return hook(prepared, cfg, seed, budget, **job)

    def architecture_checks(self, problem: Problem, prepared: Path, cfg: dict) -> list[Check]:
        hook = getattr(problem, f"checks_{self.id}", None)
        if hook is not None:
            return hook(prepared, cfg)
        return [Check("architecture checks", "architecture", f"{problem.id}/{self.id}", None,
                      detail="no architecture checks declared for this pair")]


def register_problem(problem_id: str):
    def deco(cls):
        cls.id = problem_id
        _PROBLEMS[problem_id] = cls
        return cls
    return deco


def register_method(method_id: str):
    def deco(cls):
        cls.id = method_id
        _METHODS[method_id] = cls
        return cls
    return deco


def _builtins():
    from . import builtin  # noqa: F401  (registers the built-in problems and methods)


def get_problem(p) -> Problem:
    if isinstance(p, Problem):
        return p
    _builtins()
    if p not in _PROBLEMS:
        raise UnknownName(f"unknown benchmark problem {p!r}",
                          fix=f"call pp.benchmark.problems() and use one of {sorted(_PROBLEMS)}")
    return _PROBLEMS[p]()


def get_method(m) -> Method:
    if isinstance(m, Method):
        return m
    _builtins()
    if m not in _METHODS:
        raise UnknownName(f"unknown benchmark method {m!r}",
                          fix=f"call pp.benchmark.methods() and use one of {sorted(_METHODS)}")
    return _METHODS[m]()


def problems() -> list[str]:
    _builtins()
    return sorted(_PROBLEMS)


def methods() -> list[str]:
    _builtins()
    return sorted(_METHODS)
