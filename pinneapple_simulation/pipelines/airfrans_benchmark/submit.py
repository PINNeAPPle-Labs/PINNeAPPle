"""Running the benchmark's own participant workflow, then checking the package with ours.

The numbers that go on the leaderboard are produced by the FluidsBench checkout, never here: ``validate-entry``,
``evaluate-entry``, ``package``, ``verify-package`` and their contributor validator do the work. What this module adds is
the bookkeeping — one recorded step per command, with its exit status and wall time — and then two independent reads of
the result: the adapter's :func:`~pinneapple_arena.fluidsbench.validate_package` and a recomputation of the domain
metrics from the same predictions.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["OfficialToolchain", "Step", "cross_check_domain_metrics"]


@dataclass(frozen=True)
class Step:
    """One command of the participant workflow, as it ran."""
    name: str
    command: tuple[str, ...]
    exit_code: int
    seconds: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def as_dict(self) -> dict[str, Any]:
        return {"step": self.name, "command": list(self.command), "exit_code": self.exit_code,
                "seconds": self.seconds}


@dataclass
class OfficialToolchain:
    """The benchmark's own scripts, run from a checkout with the pinned participant interpreter.

    ``python`` is the interpreter that has ``jsonschema`` (and, for ``evaluate-entry``, the AirfRANS/VTK stack); the
    benchmark publishes that environment as ``requirements-airfrans-participant.txt`` and a Dockerfile. Nothing here
    guesses a dependency: a missing one surfaces as the tool's own error."""
    checkout: str
    python: str = sys.executable
    steps: list[Step] = field(default_factory=list)

    def _run(self, name: str, *args: str, timeout: float | None = None) -> Step:
        command = (self.python, *args)
        started = time.time()
        done = subprocess.run(command, cwd=self.checkout, capture_output=True, text=True, timeout=timeout)
        step = Step(name=name, command=command, exit_code=done.returncode, seconds=round(time.time() - started, 1))
        self.steps.append(step)
        if not step.ok:
            tail = (done.stdout or done.stderr or "").strip()[-1500:]
            raise RuntimeError(f"{name} failed with exit {done.returncode}:\n{tail}")
        return step

    def _airfrans(self, name: str, *args: str) -> Step:
        return self._run(name, os.path.join("scripts", "airfrans.py"), *args)

    # ------------------------------------------------------------------ the workflow
    def fetch_support(self, destination: str, *, archive_dir: str | None = None, offline: bool = False) -> Step:
        args = ["fetch-support", "--destination", destination]
        if archive_dir:
            args += ["--archive-dir", archive_dir]
        if offline:
            args += ["--offline"]
        return self._airfrans("fetch-support", *args)

    def validate_entry(self, entry: str, dataset_root: str, support_root: str) -> Step:
        return self._airfrans("validate-entry", entry, "--dataset-root", dataset_root, "--support-root", support_root)

    def evaluate_entry(self, entry: str, dataset_root: str, support_root: str, output: str,
                       resume: bool = True) -> Step:
        args = ["evaluate-entry", entry, "--dataset-root", dataset_root, "--support-root", support_root,
                "--output", output]
        if resume:
            args.append("--resume")
        return self._airfrans("evaluate-entry", *args)

    def package(self, evaluation_root: str, config: str, discretization_cases: str, output: str,
                checkpoint: str | None = None) -> Step:
        args = ["package", "--evaluation-root", evaluation_root, "--config", config,
                "--discretization-cases", discretization_cases, "--output", output]
        if checkpoint:
            args += ["--checkpoint", f"weights={checkpoint}"]
        return self._airfrans("package", *args)

    def verify_package(self, package: str) -> Step:
        return self._airfrans("verify-package", package)

    def contributor_validator(self, package: str) -> Step:
        return self._run("contributor-validator", os.path.join("scripts", "validate_submission.py"),
                         "--contributor-stage", package)

    def adapter_validation(self, package: str) -> dict[str, Any]:
        """The adapter's own read of the package, in this process: its identity checks plus the benchmark's validator.

        Running it in-process keeps the check independent of which interpreter the participant toolchain uses — the
        adapter has no VTK dependency, and the validator it shells out to comes from the same checkout."""
        from pinneapple_arena.fluidsbench import Checkout, validate_package

        checkout = Checkout(self.checkout)
        report = validate_package(package, contract=checkout.dataset("airfrans"), checkout=checkout,
                                  run_official=True)
        self.steps.append(Step(name="adapter-validation", command=("validate_package", package),
                               exit_code=0 if report.ok else 1, seconds=0.0))
        return {"ok": report.ok, "checked": list(report.checked), "problems": list(report.problems),
                "official": dict(report.official or {})}


def cross_check_domain_metrics(entry: str, package: str, support_manifest: str, archive_root: str, *,
                               case_set: str = "standard") -> dict[str, Any]:
    """Recompute the domain-support metrics from the submitted predictions and the benchmark-owned support.

    The domain support is the native internal point cloud in its native order, so the mapping is the identity and the
    comparison with the package's own values is a genuine check of the evaluator, not of a resampling."""
    import numpy as np

    from pinneapple_arena.fluidsbench import (
        aggregate,
        load_support_release,
        map_to_support,
        score_support,
    )

    release = load_support_release(support_manifest, case_set)
    scores = []
    for case_id in release.case_ids:
        prediction = np.load(os.path.join(entry, "predictions", f"{case_id}.npz"))
        domain = release.support(case_id, "two-dimensional-domain-native", archive_root=archive_root)
        native = np.asarray(prediction["internal_points"], float)
        if native.shape[0] != domain.entity_count or np.abs(native[:, :2] - domain.coordinates).max() > 1e-9:
            raise ValueError(f"{case_id}: the predictions are not on the support entities in their native order")
        fields = {"velocity_x_pred": np.asarray(prediction["velocity"])[:, 0],
                  "velocity_y_pred": np.asarray(prediction["velocity"])[:, 1],
                  "pressure_pred": np.asarray(prediction["pressure"]).reshape(-1)}
        mapped = map_to_support(domain, points=native, fields=fields, method="identity")
        scores.append(score_support(domain, mapped.values, coverage=mapped.coverage.as_dict()))
    ours = aggregate(scores)
    with open(os.path.join(package, "metrics", "cases.json")) as handle:
        official = json.load(handle).get("metric_values") or {}
    agreement = {}
    for metric_id, value in ours.items():
        their = official.get(metric_id)
        agreement[metric_id] = None if their is None else abs(value - their) / max(abs(their), 1e-12)
    return {"ours": ours, "official": {k: official.get(k) for k in ours}, "relative_difference": agreement,
            "case_count": len(release.case_ids)}
