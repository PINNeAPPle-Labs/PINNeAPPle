"""Reproducibility: the settings a run needs, the environment it ran in, and the check that it did.

A number is reproducible when a second run of the same code, on the same machine,
with the same seed, gives the same result — or when the difference is measured
and written down. Two things make that fail quietly: a seed that was never set
for one of the three random generators (Python, NumPy, PyTorch), and an
environment that changed without anybody noticing.

``deterministic`` fixes the first, ``environment_fingerprint`` records the
second, and ``run_twice`` measures the difference instead of assuming there is
none. None of them guesses: a run that is not reproducible is reported as such,
with the fields that differ.

The fingerprint is JSON-able on purpose, so it travels inside a run record and
can be compared six months later. It records the settings that were actually in
force, not the ones the caller wishes were.

Example
-------
>>> from pinneapple_core.reproducibility import deterministic, run_twice
>>> with deterministic(seed=0):
...     report = run_twice(train_one_step)
>>> report.identical
True
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "deterministic",
    "environment_fingerprint",
    "seed_everything",
    "run_twice",
    "ReproducibilityReport",
    "compare_results",
    "fingerprint_json",
]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _torch():
    """The torch module, or None when it is not installed."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch is required for training, not for this module
        return None
    return torch


def seed_everything(seed: int = 0) -> dict[str, Any]:
    """Seed Python, NumPy and PyTorch, and return what was set.

    The returned record is meant to go into the run record: it is the evidence
    that the seed was applied, not a claim that it was.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    record: dict[str, Any] = {
        "seed": seed,
        "python_random": True,
        "numpy_random": True,
        "torch": False,
        "cuda": False,
    }
    torch = _torch()
    if torch is not None:
        torch.manual_seed(seed)
        record["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            record["cuda"] = True
    return record


@contextmanager
def deterministic(seed: int = 0, *, strict: bool = True, warn_only: bool = False):
    """Run the block with the seeds set and the deterministic algorithms on.

    ``strict`` turns on ``torch.use_deterministic_algorithms``, which makes an
    operation without a deterministic implementation raise instead of silently
    returning a different answer. ``warn_only`` keeps that as a warning — the
    honest middle ground when a model uses an operation PyTorch cannot make
    deterministic, and the caller needs the run to finish anyway.

    Everything is restored on exit, so a deterministic block inside a test does
    not leave the process in a state that changes the next test.
    """
    torch = _torch()
    previous = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "deterministic": None,
        "deterministic_warn_only": None,
        "cudnn_benchmark": None,
        "cudnn_deterministic": None,
        "cublas_env": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch is not None:
        previous["deterministic"] = torch.are_deterministic_algorithms_enabled()
        previous["deterministic_warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
        previous["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        previous["cudnn_deterministic"] = torch.backends.cudnn.deterministic

    applied = seed_everything(seed)
    if torch is not None:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if strict:
            # CUDA needs this or a matmul may pick a different reduction order.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        applied["deterministic_algorithms"] = bool(strict)
        applied["warn_only"] = bool(warn_only)
        applied["cudnn_benchmark"] = False
        applied["cudnn_deterministic"] = True
    try:
        yield applied
    finally:
        random.setstate(previous["python"])
        np.random.set_state(previous["numpy"])
        if torch is not None:
            if previous["deterministic"] is not None:
                torch.use_deterministic_algorithms(
                    previous["deterministic"],
                    warn_only=bool(previous["deterministic_warn_only"]),
                )
            torch.backends.cudnn.benchmark = bool(previous["cudnn_benchmark"])
            torch.backends.cudnn.deterministic = bool(previous["cudnn_deterministic"])
        if previous["cublas_env"] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous["cublas_env"]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _gpu_names(torch) -> list[str]:
    if torch is None or not torch.cuda.is_available():
        return []
    try:
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # pragma: no cover - driver reporting can fail on a broken install
        return []


def environment_fingerprint() -> dict[str, Any]:
    """The environment a run happened in, as JSON-able values.

    Only what the process can actually read: versions of the packages that
    change numerical results, the platform, the CPU count and the GPU names.
    A field the interpreter does not expose is ``None``, never a guess.
    """
    import matplotlib

    fingerprint: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpus": [],
        "threads": {
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }
    torch = _torch()
    if torch is not None:
        fingerprint["torch"] = torch.__version__
        fingerprint["cuda_available"] = bool(torch.cuda.is_available())
        fingerprint["cuda_version"] = getattr(torch.version, "cuda", None)
        fingerprint["gpus"] = _gpu_names(torch)
        fingerprint["determinism"].update({
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "threads": torch.get_num_threads(),
        })
    return fingerprint


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def _as_array(value) -> np.ndarray | None:
    """The numeric content of a result, or None when it has none."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, np.floating, np.integer)):
        return np.asarray(value, dtype=float)
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:  # pragma: no cover
        pass
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        parts = [_as_array(v) for v in value.values()]
        if any(p is None for p in parts):
            return None
        return np.concatenate([np.ravel(p) for p in parts]) if parts else np.asarray([])
    if isinstance(value, (list, tuple)):
        parts = [_as_array(v) for v in value]
        if any(p is None for p in parts):
            return None
        return np.concatenate([np.ravel(p) for p in parts]) if parts else np.asarray([])
    return None


def compare_results(first, second, *, atol: float = 0.0, rtol: float = 0.0) -> dict[str, Any]:
    """Compare two results, returning the verdict and the worst difference.

    Numeric content is compared with ``atol``/``rtol``; anything else (labels,
    strings, objects without numbers) only has to be equal to itself.
    """
    a, b = _as_array(first), _as_array(second)
    if a is None or b is None:
        equal = first == second
        return {"comparable": False, "identical": bool(equal), "max_abs_diff": None,
                "max_rel_diff": None, "atol": atol, "rtol": rtol}
    if a.shape != b.shape:
        return {"comparable": True, "identical": False, "max_abs_diff": None,
                "max_rel_diff": None, "atol": atol, "rtol": rtol,
                "reason": f"shape changed from {a.shape} to {b.shape}"}
    if a.size == 0:
        return {"comparable": True, "identical": True, "max_abs_diff": 0.0,
                "max_rel_diff": 0.0, "atol": atol, "rtol": rtol}
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, diff / scale, 0.0)
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    max_rel = float(np.nanmax(rel)) if rel.size else 0.0
    identical = bool(np.all(diff <= atol + rtol * scale))
    return {"comparable": True, "identical": identical, "max_abs_diff": max_abs,
            "max_rel_diff": max_rel, "atol": atol, "rtol": rtol}


@dataclass
class ReproducibilityReport:
    """What two runs of the same callable had to say about each other."""

    identical: bool
    seed: int | None
    environment: dict[str, Any] = field(default_factory=dict)
    differences: list[dict[str, Any]] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    scored: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "identical": self.identical,
            "seed": self.seed,
            "scored": self.scored,
            "environment": self.environment,
            "differences": self.differences,
            "checked": self.checked,
            "note": self.note,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Reproducibility check",
            "",
            f"- verdict: **{'reproduced' if self.identical else 'divergent'}**",
            f"- seed: `{self.seed}`",
            f"- environment: Python {self.environment.get('python')}, "
            f"NumPy {self.environment.get('numpy')}, torch {self.environment.get('torch')}",
        ]
        if not self.identical:
            lines.append("")
            lines.append("| field | max abs diff | max rel diff | atol | rtol |")
            lines.append("|---|---|---|---|---|")
            for diff in self.differences:
                lines.append(
                    f"| {diff.get('field', '?')} | {diff.get('max_abs_diff')} | "
                    f"{diff.get('max_rel_diff')} | {diff.get('atol')} | {diff.get('rtol')} |"
                )
        if self.note:
            lines.append("")
            lines.append(self.note)
        return "\n".join(lines) + "\n"


def _split(result) -> dict[str, Any]:
    """A result as named fields, so a difference says which field moved."""
    if isinstance(result, Mapping):
        return dict(result)
    return {"result": result}


def run_twice(
    fn: Callable[[], Any],
    *,
    seed: int = 0,
    atol: float = 0.0,
    rtol: float = 0.0,
    strict: bool = True,
    fields: Sequence[str] | None = None,
) -> ReproducibilityReport:
    """Call ``fn`` twice under the same seed and report whether the results match.

    ``fields`` narrows the comparison to the named keys when ``fn`` returns a
    mapping; by default every key is compared. A field that cannot be compared
    numerically is reported as not scored, never silently counted as equal.
    """
    environment = environment_fingerprint()
    with deterministic(seed=seed, strict=strict):
        first = fn()
    with deterministic(seed=seed, strict=strict):
        second = fn()

    a, b = _split(first), _split(second)
    keys = list(fields) if fields is not None else sorted(set(a) | set(b))
    differences: list[dict[str, Any]] = []
    checked: list[str] = []
    for key in keys:
        if key not in a or key not in b:
            differences.append({"field": key, "max_abs_diff": None, "max_rel_diff": None,
                                "atol": None, "rtol": None, "reason": "missing in one run"})
            continue
        outcome = dict(compare_results(a[key], b[key], atol=atol, rtol=rtol))
        outcome["field"] = key
        if not outcome["comparable"]:
            outcome["reason"] = "not comparable numerically"
        else:
            checked.append(key)
        if not outcome["identical"]:
            differences.append(outcome)

    note = ""
    if not checked:
        note = "No field could be compared numerically; the verdict does not claim reproducibility."
    return ReproducibilityReport(
        identical=bool(not differences) and bool(checked),
        seed=seed,
        environment=environment,
        differences=differences,
        checked=checked,
        scored=bool(checked),
        note=note,
    )


def fingerprint_json(fingerprint: dict[str, Any] | None = None, indent: int = 2) -> str:
    """The fingerprint as JSON, for a run record."""
    return json.dumps(fingerprint if fingerprint is not None else environment_fingerprint(),
                      indent=indent, sort_keys=True, default=str)
