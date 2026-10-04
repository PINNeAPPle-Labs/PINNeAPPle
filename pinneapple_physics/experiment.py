"""``Experiment``: one reproducible run of a method on a problem, with its metrics and provenance.

>>> import pinneapple as pp
>>> exp = pp.Experiment("burgers_1d", method="pinn", options={"epochs": 3000}, reference="analytic", seed=1)
>>> result = exp.run()
>>> result.metrics["relative_l2"]["u"], result.wall_time_s
>>> result.save("runs/burgers_pinn")          # JSON record (+ model weights when there is a model)

A run fixes every random source it controls (Python, NumPy, PyTorch, and PyTorch's deterministic-algorithms mode
while it runs), records the problem fingerprint, the library and dependency versions and the machine, and scores
the solution against ``reference`` when one is given. Running the same experiment twice on the same machine gives
the same predictions (``tests/test_solve_compare_experiment.py`` checks it bit for bit on CPU).
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from . import metrics as _metrics
from .physical_problem import PhysicalProblem
from .solving import Solution, as_problem, solve

__all__ = ["Experiment", "ExperimentResult"]


@contextlib.contextmanager
def _seeded(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover
        yield
        return
    torch.manual_seed(seed)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _environment() -> Dict[str, Any]:
    env = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
           "cpu_count": os.cpu_count(), "numpy": np.__version__}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:  # pragma: no cover
        pass
    try:
        import pinneapple
        env["pinneapple"] = pinneapple.__version__
    except Exception:  # pragma: no cover
        pass
    return env


@dataclass
class ExperimentResult:
    config: Dict[str, Any]
    problem_fingerprint: str
    solution: Solution
    metrics: Optional[Dict[str, Dict[str, float]]]
    wall_time_s: float
    started_at: str
    environment: Dict[str, Any]
    n_eval_points: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"config": self.config, "problem_fingerprint": self.problem_fingerprint, "metrics": self.metrics,
                "wall_time_s": self.wall_time_s, "started_at": self.started_at, "environment": self.environment,
                "n_eval_points": self.n_eval_points, "method_info": self.solution.info,
                "problem": self.solution.problem.to_dict()}

    def save(self, directory: str) -> str:
        """Write ``result.json`` (and ``model.pt`` when the solution has a PyTorch model) to ``directory``."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "result.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        model = self.solution.model
        if model is not None and hasattr(model, "state_dict"):
            import torch
            torch.save(model.state_dict(), os.path.join(directory, "model.pt"))
        return path

    @staticmethod
    def load_record(directory: str) -> Dict[str, Any]:
        """The saved JSON record (the solution object itself is not rebuilt; reload weights with ``torch.load``)."""
        with open(os.path.join(directory, "result.json")) as f:
            return json.load(f)


@dataclass
class Experiment:
    """A method applied to a problem, with a seed, options and an optional reference to score against."""
    problem: Union[PhysicalProblem, Any, str]
    method: str = "pinn"
    options: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    reference: Optional[str] = None
    reference_options: Dict[str, Any] = field(default_factory=dict)
    n_eval: int = 4096
    metric_names: Sequence[str] = ("relative_l2", "rmse", "max_abs")
    name: str = ""

    def __post_init__(self):
        self.problem = as_problem(self.problem)
        if not self.name:
            self.name = f"{self.problem.name}/{self.method}/seed{self.seed}"

    def config(self) -> Dict[str, Any]:
        return {"name": self.name, "problem": self.problem.name, "method": self.method, "options": dict(self.options),
                "seed": self.seed, "reference": self.reference, "reference_options": dict(self.reference_options),
                "n_eval": self.n_eval, "metric_names": list(self.metric_names)}

    def evaluation_points(self, reference: Optional[Solution]) -> Optional[np.ndarray]:
        rng = np.random.default_rng(self.seed + 1_000_003)
        grid = reference.grid_points() if reference is not None else None
        if grid is not None:
            return grid if len(grid) <= self.n_eval else grid[rng.choice(len(grid), self.n_eval, replace=False)]
        b = self.problem.domain_bounds
        if not b or set(b) != set(self.problem.coords):
            return None
        lo = np.array([b[c][0] for c in self.problem.coords])
        hi = np.array([b[c][1] for c in self.problem.coords])
        return lo + (hi - lo) * rng.random((self.n_eval, len(self.problem.coords)))

    def run(self) -> ExperimentResult:
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        opts = dict(self.options)
        opts.setdefault("seed", self.seed)
        t0 = time.perf_counter()
        with _seeded(self.seed):
            sol = solve(self.problem, self.method, **opts)
        elapsed = time.perf_counter() - t0
        scores, n_pts = None, 0
        if self.reference is not None:
            ref = solve(self.problem, self.reference, **dict(self.reference_options))
            pts = self.evaluation_points(ref)
            if pts is None:
                raise ValueError("cannot choose evaluation points: the reference is not a grid solution and the "
                                 "problem has no box bounds for every coordinate")
            scores = _metrics.summary(sol.predict(pts), ref.predict(pts), self.problem.fields, self.metric_names)
            n_pts = len(pts)
        return ExperimentResult(config=self.config(), problem_fingerprint=self.problem.fingerprint(), solution=sol,
                                metrics=scores, wall_time_s=elapsed, started_at=started, environment=_environment(),
                                n_eval_points=n_pts)


def run_all(experiments: Sequence[Experiment]) -> Dict[str, ExperimentResult]:
    """Run several experiments; keyed by experiment name."""
    return {e.name: e.run() for e in experiments}
