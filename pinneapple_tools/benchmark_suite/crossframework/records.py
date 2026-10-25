"""Run records of the cross-framework benchmark: environment, threads, load, time caps, the result file schema.

A runner calls :func:`setup_threads` before importing its framework, measures its training with :class:`Clock`
(wall and process CPU time, load average before and after), stops at a cap with :class:`TimeCap`, and writes its
exports with :func:`write_run`. The driver validates every result against :data:`RESULT_FIELDS`.
"""
from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["setup_threads", "loadavg", "cpu_model", "package_versions", "environment", "Clock", "TimeCap",
           "write_run", "read_run", "RESULT_FIELDS", "validate_result", "FRAMEWORK_PACKAGES"]

#: distribution names of each framework (the versions go into every result)
FRAMEWORK_PACKAGES = {
    "pinneapple": ["pinneapple"],
    "pinneapple_fixed": ["pinneapple"],
    "pinneapple_equal": ["pinneapple"],
    "deepxde": ["deepxde"],
    "physicsnemo_sym": ["nvidia-physicsnemo.sym", "nvidia-physicsnemo"],
    "physicsnemo": ["nvidia-physicsnemo"],
    "neuraloperator": ["neuraloperator"],
}

#: fields every result JSON has (the driver refuses a result without them)
RESULT_FIELDS = ("track", "problem", "framework", "mode", "seed", "status", "iterations", "wall_s", "cpu_s",
                 "loadavg_before", "loadavg_after", "threads", "versions", "config", "source")


def setup_threads(n: int) -> None:
    """Pin the thread count of this process (call before importing torch / a framework): OMP / MKL / OpenBLAS
    environment variables and ``torch.set_num_threads``."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    import torch
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:            # already set (interop threads can be set once per process)
        pass


def loadavg() -> list:
    try:
        return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except OSError:
        return [float(x) for x in os.getloadavg()]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def package_versions(names) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def environment(threads: int) -> dict[str, Any]:
    """Machine and library record stored next to the results."""
    import torch
    return {"cpu": cpu_model(), "n_cpus": os.cpu_count(), "threads": threads, "python": platform.python_version(),
            "platform": platform.platform(), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "cuda_available": bool(torch.cuda.is_available()),
            "numpy": np.__version__, "loadavg": loadavg(),
            "versions": package_versions(sorted({p for v in FRAMEWORK_PACKAGES.values() for p in v}))}


@dataclass
class Clock:
    """Wall and process CPU time of a block, with the load average before and after."""
    wall_s: float = 0.0
    cpu_s: float = 0.0
    loadavg_before: list = field(default_factory=list)
    loadavg_after: list = field(default_factory=list)

    def __enter__(self) -> Clock:
        self.loadavg_before = loadavg()
        self._w, self._c = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *exc) -> None:
        self.wall_s = time.perf_counter() - self._w
        self.cpu_s = time.process_time() - self._c
        self.loadavg_after = loadavg()


class TimeCap:
    """A per-run time budget, measured as process CPU time (``kind="cpu"``) or wall time (``kind="wall"``).

    With one thread per run and a quiet machine the two are equal; on a shared machine the CPU time is the budget
    that does not depend on what the other processes do. ``expired()`` is cheap enough to call every iteration.
    """

    def __init__(self, seconds: float, kind: str = "cpu"):
        if kind not in ("cpu", "wall"):
            raise ValueError("kind must be 'cpu' or 'wall'")
        self.seconds, self.kind = float(seconds), kind
        self._now = time.process_time if kind == "cpu" else time.perf_counter
        self._t0 = self._now()

    def restart(self) -> None:
        self._t0 = self._now()

    def used(self) -> float:
        return self._now() - self._t0

    def expired(self) -> bool:
        return self.used() >= self.seconds


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_run(out_dir, run_id: str, meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> Path:
    """Write ``<run_id>.json`` (metadata) and ``<run_id>.npz`` (exported predictions) into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{run_id}.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    p = out / f"{run_id}.json"
    p.write_text(json.dumps(dict(meta), indent=1, default=_json_default))
    return p


def read_run(out_dir, run_id: str):
    out = Path(out_dir)
    meta = json.loads((out / f"{run_id}.json").read_text())
    arrays = {}
    if (out / f"{run_id}.npz").exists():
        with np.load(out / f"{run_id}.npz") as z:
            arrays = {k: z[k] for k in z.files}
    return meta, arrays


def validate_result(meta: Mapping[str, Any]) -> list:
    """Missing schema fields of a result (empty when valid)."""
    return [f for f in RESULT_FIELDS if f not in meta]
