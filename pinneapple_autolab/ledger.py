"""The evidence ledger: every simulation the lab ran (why, with what, what it cost, what came out) and every event
of the session, in the order they happened.

The ledger is the only source the report, the dashboard and an agent may read numbers from: a value that is not in a
:class:`RunRecord` was not measured. ``Ledger.event`` appends to ``events.jsonl`` as it goes (a dashboard can tail
it); ``Ledger.save`` writes ``ledger.json`` with the runs, the environment and the budget.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = ["RunRecord", "Ledger", "environment", "BudgetExhaustedError"]

PURPOSES = ("baseline", "doe", "acquisition", "verification", "probe")


class BudgetExhaustedError(RuntimeError):
    """The mission's simulation budget has been spent."""


@dataclass
class RunRecord:
    """One reference simulation. ``outputs`` holds only what the solver produced (empty when it failed);
    ``predicted`` what the surrogate said before the run (for the held-out error), ``reason`` why it was run."""
    id: str
    purpose: str
    params: dict[str, float]
    ok: bool
    outputs: dict[str, float] = field(default_factory=dict)
    message: str = ""
    cost_s: float = 0.0
    started: str = ""
    reason: str = ""
    predicted: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    fields_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001 - a missing tool is recorded as unknown
        return ""


def environment(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Hardware and software of this machine: CPU, cores, memory, OS, Python and package versions, git commit."""
    import importlib
    env: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(),
                           "cpu_count": os.cpu_count(), "machine": platform.machine()}
    try:
        with open("/proc/cpuinfo") as f:
            env["cpu"] = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "")
        with open("/proc/meminfo") as f:
            env["mem_gb"] = round(int(f.readline().split()[1]) / 1024 ** 2, 1)
    except OSError:
        pass
    for mod in ("numpy", "scipy", "sklearn", "torch", "pinneapple"):
        try:
            env[mod] = getattr(importlib.import_module(mod), "__version__", "")
        except Exception:  # noqa: BLE001
            env[mod] = None
    env["git_commit"] = _cmd(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"])
    env.update(extra or {})
    return env


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


class Ledger:
    """Runs and events of one lab session, kept in ``folder``. ``listeners``: callables ``f(event_dict)`` called
    on every event (a live dashboard, a test)."""

    def __init__(self, folder: str, max_runs: int | None = None, max_wall_s: float | None = None,
                 listeners: list[Callable[[dict], None]] | None = None):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.runs: list[RunRecord] = []
        self.max_runs, self.max_wall_s = max_runs, max_wall_s
        self.listeners = list(listeners or [])
        self.env = environment()
        self._t0 = time.time()
        self._events = os.path.join(folder, "events.jsonl")
        open(self._events, "w").close()

    # ------------------------------------------------------------------ budget
    @property
    def used_runs(self) -> int:
        return len(self.runs)

    @property
    def used_wall_s(self) -> float:
        return float(sum(r.cost_s for r in self.runs))

    def remaining(self) -> int:
        left = (self.max_runs - self.used_runs) if self.max_runs is not None else 10 ** 9
        if self.max_wall_s is not None and self.used_wall_s >= self.max_wall_s:
            left = 0
        return max(0, left)

    def charge(self) -> None:
        """Raise :class:`BudgetExhaustedError` if no simulation is left."""
        if self.remaining() <= 0:
            raise BudgetExhaustedError(f"simulation budget spent ({self.used_runs} runs, {self.used_wall_s:.0f} s)")

    # ------------------------------------------------------------------ records
    def add(self, rec: RunRecord) -> RunRecord:
        self.runs.append(rec)
        self.event("run", f"{rec.id} ({rec.purpose}): " + ("ok" if rec.ok else f"FAILED: {rec.message}"),
                   run=rec.to_dict())
        return rec

    def event(self, stage: str, message: str, **data: Any) -> dict[str, Any]:
        ev = {"t": round(time.time() - self._t0, 2), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage,
              "message": message, **_jsonable(data)}
        with open(self._events, "a") as f:
            f.write(json.dumps(ev) + "\n")
        for fn in self.listeners:
            fn(ev)
        return ev

    def ok_runs(self, probes: bool = False) -> list[RunRecord]:
        """Successful runs; ``probes=False`` leaves out the "probe" runs (another fidelity, e.g. a finer mesh,
        that must not be mixed with the training data)."""
        return [r for r in self.runs if r.ok and (probes or r.purpose != "probe")]

    def arrays(self, names: list[str], outputs: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Design matrix (columns ``names``) and outputs of the successful runs."""
        ok = self.ok_runs()
        X = np.array([[r.params[k] for k in names] for r in ok], float).reshape(len(ok), len(names))
        return X, {k: np.array([r.outputs.get(k, np.nan) for r in ok], float) for k in outputs}

    def next_id(self, purpose: str) -> str:
        return f"run{len(self.runs) + 1:03d}-{purpose}"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({"environment": self.env, "budget": {"max_runs": self.max_runs, "max_wall_s": self.max_wall_s,
                                                               "used_runs": self.used_runs,
                                                               "used_wall_s": round(self.used_wall_s, 1)},
                          "runs": [r.to_dict() for r in self.runs]})

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(self.folder, "ledger.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> Ledger:
        d = json.load(open(path))
        led = cls.__new__(cls)
        led.folder = os.path.dirname(path)
        led.runs = [RunRecord(**r) for r in d["runs"]]
        led.max_runs, led.max_wall_s = d["budget"]["max_runs"], d["budget"]["max_wall_s"]
        led.listeners, led.env, led._t0 = [], d["environment"], time.time()
        led._events = os.path.join(led.folder, "events.jsonl")
        return led
