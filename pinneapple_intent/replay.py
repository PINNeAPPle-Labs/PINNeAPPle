"""Rebuild a study from the files it left behind, with no agent and no sentence.

The point of the contract is this module. If a run cannot be reproduced from ``contract.json`` alone, the contract is
missing something; if it reproduces to different numbers, either the library changed or the run was not
deterministic, and both are worth knowing before anyone acts on the result.

    python -m pinneapple_intent replay runs/heat_wall-1a03dab1

A replay compares the quantity of interest and every rung's outputs with the recorded values, and reports the
differences it found instead of a single "ok".
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .contract import RUN_SCHEMA_ID, SCHEMA_ID, StudyContract
from .execute import StudyRun, run_study

# Two runs of the same contract may legitimately differ in the last bits (a solver's thread count, a BLAS kernel);
# anything larger than this is reported as a difference rather than noise.
TOLERANCE = 1e-9


@dataclass
class ReplayResult:
    """What a replay found: the new run, the recorded one, and every number that did not come back the same."""
    run: StudyRun
    recorded: dict[str, Any]
    differences: list[dict[str, Any]] = field(default_factory=list)
    integrity: list[str] = field(default_factory=list)

    @property
    def matches(self) -> bool:
        return not self.differences and not self.integrity

    def to_dict(self) -> dict[str, Any]:
        return {"matches": self.matches, "contract_id": self.run.contract.id,
                "decision": self.run.decision, "recorded_decision": self.recorded.get("decision"),
                "differences": self.differences, "integrity": self.integrity}

    def summary(self) -> str:
        if self.matches:
            return (f"the study replays exactly: {self.run.contract.id}, decision {self.run.decision}, "
                    f"{len(self.run.measurements)} rungs")
        parts = []
        if self.integrity:
            parts.append("integrity: " + "; ".join(self.integrity))
        if self.differences:
            parts.append(f"{len(self.differences)} numbers differ")
        return f"the replay does not match the record ({', '.join(parts)})"


def replay(path: str, *, out_dir: str | None = None, strict: bool = False) -> ReplayResult:
    """Re-run the study recorded in ``path`` (a run directory or its ``run.json``) and compare it with the record.

    ``strict`` raises when the replay does not match; by default the differences are returned so a caller can report
    them with the numbers attached, which is more useful than an exception.
    """
    folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    run_file = os.path.join(folder, "run.json") if os.path.isdir(path) else path
    contract_file = os.path.join(folder, "contract.json")
    with open(run_file, encoding="utf-8") as handle:
        recorded = json.load(handle)
    if recorded.get("schema") != RUN_SCHEMA_ID:
        raise ValueError(f"not a {RUN_SCHEMA_ID} file: {run_file}")

    integrity: list[str] = []
    if os.path.exists(contract_file):
        contract = StudyContract.read(contract_file)
        if contract.id != recorded.get("contract_id"):
            integrity.append(f"contract.json hashes to {contract.id} but the run recorded "
                             f"{recorded.get('contract_id')}: the contract was edited after the run")
    else:
        contract = StudyContract.from_dict(recorded["contract"])
        integrity.append("contract.json was missing; the contract was read from run.json, so a change to it cannot "
                         "be told from a change to the library")

    rungs = tuple(stage["rung"] for stage in recorded.get("stages") or () if stage.get("ran")) or None
    fresh = run_study(contract, out_dir=out_dir, rungs=rungs)
    differences = _compare(recorded, fresh, contract)
    result = ReplayResult(run=fresh, recorded=recorded, differences=differences, integrity=integrity)
    if strict and not result.matches:
        raise RuntimeError(result.summary())
    if out_dir:
        _write(os.path.join(out_dir, "replay.json"), json.dumps(result.to_dict(), indent=2))
    return result


def _compare(recorded: dict[str, Any], fresh: StudyRun, contract: StudyContract) -> list[dict[str, Any]]:
    """Every recorded output against the fresh one, with the same tolerance for each pair."""
    differences: list[dict[str, Any]] = []
    old = {stage["rung"]: stage for stage in recorded.get("stages") or ()}
    for name, measurement in fresh.measurements.items():
        before = old.get(name)
        if before is None:
            differences.append({"rung": name, "quantity": "(stage)", "recorded": None, "replayed": "ran",
                                "reason": "the recorded run has no such rung"})
            continue
        if bool(before.get("ran")) != measurement.ran:
            differences.append({"rung": name, "quantity": "(stage)", "recorded": before.get("ran"),
                                "replayed": measurement.ran,
                                "reason": f"recorded reason: {before.get('reason') or 'ran'}; "
                                          f"now: {measurement.reason or 'ran'}"})
            continue
        for quantity, value in measurement.outputs.items():
            previous = (before.get("outputs") or {}).get(quantity)
            if previous is None:
                continue
            delta = _relative(previous, value)
            if delta is not None and delta > TOLERANCE:
                differences.append({"rung": name, "quantity": quantity, "recorded": previous, "replayed": value,
                                    "relative_difference": delta})
    if recorded.get("decision") != fresh.decision:
        differences.append({"rung": "(all)", "quantity": "decision", "recorded": recorded.get("decision"),
                            "replayed": fresh.decision,
                            "reason": "the checks or the rungs that ran changed"})
    return differences


def _relative(previous: Any, value: Any) -> float | None:
    try:
        a, b = float(previous), float(value)
    except (TypeError, ValueError):
        return None if previous == value else float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / scale


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


__all__ = ["ReplayResult", "replay", "TOLERANCE", "SCHEMA_ID"]
