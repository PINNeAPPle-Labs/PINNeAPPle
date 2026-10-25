"""
The result contract every model runner (inference *and* fine-tune) returns.
Ported from the original suite_core.py with one addition: `ModelResult` now
also carries an optional `metrics["loss_curve"]` for fine-tune runs and a
`job_kind` field ("infer" | "finetune" | "benchmark") set by the caller.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_NOT_INSTALLED = "not_installed"
STATUS_MANUAL_INSTALL = "manual_install"
STATUS_SKIPPED_HEAVY = "skipped_heavy"


@dataclass
class ModelResult:
    name: str
    slug: str
    category: str
    domains: list[str]
    repo: str
    status: str
    dataset_name: str = ""
    dataset_description: str = ""
    input_shape: str | None = None
    output_shape: str | None = None
    output_summary: dict | None = None
    metrics: dict = field(default_factory=dict)
    error: str | None = None
    note: str | None = None
    local_command: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_array(arr) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "n": int(a.size),
    }


def not_installed(name: str, slug: str, category: str, domains: list[str], repo: str,
                   pip_hint: str, dataset: dict | None = None,
                   local_command: str | None = None) -> ModelResult:
    return ModelResult(
        name=name, slug=slug, category=category, domains=domains, repo=repo,
        status=STATUS_NOT_INSTALLED,
        dataset_name=dataset["name"] if dataset else "",
        dataset_description=dataset["description"] if dataset else "",
        note=f"Not installed. Try: {pip_hint}",
        local_command=local_command,
    )


def manual_install(name: str, slug: str, category: str, domains: list[str], repo: str,
                    instructions: str, dataset: dict | None = None,
                    local_command: str | None = None) -> ModelResult:
    return ModelResult(
        name=name, slug=slug, category=category, domains=domains, repo=repo,
        status=STATUS_MANUAL_INSTALL,
        dataset_name=dataset["name"] if dataset else "",
        dataset_description=dataset["description"] if dataset else "",
        note=instructions,
        local_command=local_command,
    )


def skipped_heavy(name: str, slug: str, category: str, domains: list[str], repo: str,
                   pip_hint: str, dataset: dict | None = None,
                   local_command: str | None = None) -> ModelResult:
    return ModelResult(
        name=name, slug=slug, category=category, domains=domains, repo=repo,
        status=STATUS_SKIPPED_HEAVY,
        dataset_name=dataset["name"] if dataset else "",
        dataset_description=dataset["description"] if dataset else "",
        note=f"Heavy/GPU-oriented dependency — rerun with include_heavy=true ({pip_hint})",
        local_command=local_command,
    )


def failed(name: str, slug: str, category: str, domains: list[str], repo: str,
           exc: Exception, duration: float, dataset: dict | None = None) -> ModelResult:
    tb_line = traceback.format_exc().strip().splitlines()[-1]
    return ModelResult(
        name=name, slug=slug, category=category, domains=domains, repo=repo,
        status=STATUS_FAILED,
        dataset_name=dataset["name"] if dataset else "",
        dataset_description=dataset["description"] if dataset else "",
        metrics={"inference_seconds": round(duration, 4)},
        error=f"{exc.__class__.__name__}: {tb_line}",
    )


def ok(name: str, slug: str, category: str, domains: list[str], repo: str,
       dataset: dict, duration: float, output_shape: str | None = None,
       output_summary: dict | None = None, note: str | None = None,
       input_shape: str | None = None, metrics: dict | None = None) -> ModelResult:
    m = {"inference_seconds": round(duration, 4)}
    if metrics:
        m.update(metrics)
    return ModelResult(
        name=name, slug=slug, category=category, domains=domains, repo=repo,
        status=STATUS_OK,
        dataset_name=dataset["name"], dataset_description=dataset["description"],
        input_shape=input_shape, output_shape=output_shape, output_summary=output_summary,
        metrics=m, note=note,
    )


class Timer:
    """`with Timer() as t: ...` then read `t.elapsed` at any point, including
    from an `except` clause before the block exits."""
    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        return False

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0
