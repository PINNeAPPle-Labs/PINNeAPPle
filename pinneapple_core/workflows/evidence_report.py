"""Deterministic, provenance-aware report export for experiment runs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_evidence_report(run: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    """No implied physical approval: only explicit successful verification counts."""
    by_id = {s["step_id"]: s for s in steps}
    verify = by_id.get("verify", {})
    outputs = verify.get("outputs") or {}
    approved = verify.get("status") in ("done", "skipped") and outputs.get("approved") is True
    workflow = run.get("workflow") or {}
    report = {
        "schema_version": "1.0",
        "run_id": run["id"],
        "workflow_name": run.get("name"),
        "run_status": run.get("status"),
        "objective": (workflow.get("metadata") or {}).get("objective"),
        "verification": {
            "approved": approved,
            "evidence": outputs if approved else {"reason": outputs.get("reason", "not approved")},
        },
        "steps": [
            {"id": s["step_id"], "status": s["status"], "hash": s.get("hash"),
             "seconds": s.get("seconds"), "error": s.get("error")}
            for s in steps
        ],
        "limitations": [
            "Approval only covers the explicit verification evidence and context of use.",
            "This report does not certify safety or authorize physical equipment control.",
        ],
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), default=str).encode()
    report["sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def export_evidence_report(runner: Any, run_id: str, destination: str | Path) -> Path:
    report = build_evidence_report(runner.get_run(run_id), runner.steps(run_id))
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
